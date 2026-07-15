import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.type.VoidType;
import com.github.javaparser.ast.type.ReferenceType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class RefactorApp {

    static class RangeOffset {
        int start;
        int end;
        RangeOffset(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Uso: java -jar refactor.jar <ruta.java> <start-end> <start-end> ...");
            System.exit(1);
        }

        Path javaFilePath = Paths.get(args[0]);
        String content = Files.readString(javaFilePath);
        CompilationUnit cu = StaticJavaParser.parse(content);

        List<RangeOffset> ranges = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            String[] parts = args[i].split("-");
            ranges.add(new RangeOffset(Integer.parseInt(parts[0]), Integer.parseInt(parts[1])));
        }
        ranges.sort(Comparator.comparingInt(r -> (r.end - r.start)));

        Map<String, Integer> methodCounters = new HashMap<>();

        for (RangeOffset rangeOffset : ranges) {
            Position startPos = getPositionAtOffset(content, rangeOffset.start);
            Position endPos = getPositionAtOffset(content, rangeOffset.end);
            Range targetRange = Range.range(startPos, endPos);

            List<Statement> statementsToExtract = cu.findAll(Statement.class).stream()
                    .filter(stmt -> stmt.getRange().isPresent() && isWithin(stmt.getRange().get(), targetRange))
                    .filter(stmt -> !hasAncestorInRange(stmt, targetRange))
                    .collect(Collectors.toList());

            if (!statementsToExtract.isEmpty()) {
                Statement firstStmt = statementsToExtract.get(0);
                String parentContainerName = getContainerName(firstStmt);

                int currentCount = methodCounters.getOrDefault(parentContainerName, 1);
                methodCounters.put(parentContainerName, currentCount + 1);

                String generatedMethodName = parentContainerName + "_extraction_" + currentCount;

                extractMethod(cu, targetRange, statementsToExtract, generatedMethodName);
            }
        }

        Files.writeString(javaFilePath, cu.toString());
    }

    private static void extractMethod(CompilationUnit cu, Range targetRange, List<Statement> statementsToExtract, String newMethodName) {
        Statement firstStmt = statementsToExtract.get(0);
        Statement lastStmt = statementsToExtract.get(statementsToExtract.size() - 1);
        Node parentNode = firstStmt.getParentNode().orElseThrow();

        // --- ANÁLISIS DEL CONTENEDOR PADRE ---
        BodyDeclaration<?> parentContainer = null;
        boolean isStaticContext = false;
        List<Parameter> containerParameters = new ArrayList<>();
        NodeList<ReferenceType> containerThrownExceptions = new NodeList<>();

        Optional<MethodDeclaration> methodParent = firstStmt.findAncestor(MethodDeclaration.class);
        Optional<ConstructorDeclaration> constructorParent = firstStmt.findAncestor(ConstructorDeclaration.class);
        Optional<InitializerDeclaration> initializerParent = firstStmt.findAncestor(InitializerDeclaration.class);

        if (methodParent.isPresent()) {
            MethodDeclaration m = methodParent.get();
            parentContainer = m;
            isStaticContext = m.isStatic();
            containerParameters.addAll(m.getParameters());
            containerThrownExceptions.addAll(m.getThrownExceptions());
        } else if (constructorParent.isPresent()) {
            ConstructorDeclaration c = constructorParent.get();
            parentContainer = c;
            isStaticContext = false;
            containerParameters.addAll(c.getParameters());
            containerThrownExceptions.addAll(c.getThrownExceptions());
        } else if (initializerParent.isPresent()) {
            InitializerDeclaration i = initializerParent.get();
            parentContainer = i;
            isStaticContext = i.isStatic();
        } else {
            return;
        }

        final BodyDeclaration<?> finalParentContainer = parentContainer;

        // --- INICIO DEL ANÁLISIS DE FLUJO DE DATOS (ESTRUCTURAL / INDEPENDIENTE DE RANGOS) ---
        Map<String, Type> declaredBefore = new LinkedHashMap<>();
        Map<String, Type> declaredInside = new LinkedHashMap<>();

        containerParameters.forEach(p -> declaredBefore.put(p.getNameAsString(), p.getType()));

        finalParentContainer.findAll(VariableDeclarator.class).forEach(vd -> {
            boolean isBeforeTarget = isNodeBefore(vd, firstStmt) && !isAncestorOrSelf(vd, firstStmt);
            boolean isInsideTarget = isInsideTarget(vd, statementsToExtract);

            if (isBeforeTarget) {
                if (isVariableInScopeAtRange(vd, firstStmt)) {
                    declaredBefore.put(vd.getNameAsString(), vd.getType());
                }
            } else if (isInsideTarget) {
                if (canVariableEscape(vd, statementsToExtract)) {
                    declaredInside.put(vd.getNameAsString(), vd.getType());
                }
            }
        });

        Set<String> usedInside = new LinkedHashSet<>();
        Set<String> modifiedInside = new LinkedHashSet<>();

        for (Statement stmt : statementsToExtract) {
            stmt.findAll(NameExpr.class).forEach(ne -> usedInside.add(ne.getNameAsString()));

            stmt.findAll(AssignExpr.class).forEach(ae -> {
                if (ae.getTarget().isNameExpr()) {
                    modifiedInside.add(ae.getTarget().asNameExpr().getNameAsString());
                }
            });

            stmt.findAll(UnaryExpr.class).forEach(ue -> {
                UnaryExpr.Operator op = ue.getOperator();
                if ((op == UnaryExpr.Operator.PREFIX_INCREMENT || op == UnaryExpr.Operator.POSTFIX_INCREMENT ||
                     op == UnaryExpr.Operator.PREFIX_DECREMENT || op == UnaryExpr.Operator.POSTFIX_DECREMENT)
                     && ue.getExpression().isNameExpr()) {
                    modifiedInside.add(ue.getExpression().asNameExpr().getNameAsString());
                }
            });
        }

        Set<String> usedAfter = new LinkedHashSet<>();
        finalParentContainer.findAll(Statement.class).stream()
                .filter(stmt -> isNodeBefore(lastStmt, stmt) && !isAncestorOrSelf(stmt, lastStmt) && !isInsideTarget(stmt, statementsToExtract))
                .forEach(stmt -> stmt.findAll(NameExpr.class).forEach(ne -> {
                    if (!isReferenceToNewDeclaration(ne, statementsToExtract, finalParentContainer)) {
                        usedAfter.add(ne.getNameAsString());
                    }
                }));

        // --- RESOLUCIÓN DE LA FIRMA DEL MÉTODO ---
        List<String> requiredParams = new ArrayList<>();
        for (String var : usedInside) {
            if (declaredBefore.containsKey(var) && !declaredInside.containsKey(var)) {
                requiredParams.add(var);
            }
        }

        // --- RESOLUCIÓN DEL VALOR DE RETORNO (replica la validación de Eclipse) ---
        // Eclipse recopila TODAS las variables que necesitarían "escapar" del bloque
        // extraído (declaradas o modificadas dentro y usadas después). Si hay más de
        // una, Eclipse rechaza la extracción con "Ambiguous return value: selected
        // block contains more than one assignment to local variable" en vez de
        // generar una firma que solo pueda devolver un único valor. Aquí replicamos
        // exactamente ese comportamiento: si detectamos más de una variable candidata,
        // abortamos la extracción tal cual haría Eclipse, en vez de quedarnos con la
        // primera y dejar las demás "huérfanas" (lo que producía los symbol not found).
        LinkedHashMap<String, Type> candidateReturns = new LinkedHashMap<>();
        LinkedHashMap<String, Boolean> candidateIsDeclaredInside = new LinkedHashMap<>();

        for (String var : usedAfter) {
            if (modifiedInside.contains(var) && declaredBefore.containsKey(var)) {
                candidateReturns.put(var, declaredBefore.get(var));
                candidateIsDeclaredInside.put(var, false);
            } else if (declaredInside.containsKey(var)) {
                candidateReturns.put(var, declaredInside.get(var));
                candidateIsDeclaredInside.put(var, true);
            }
        }

        if (candidateReturns.size() > 1) {
            System.err.println("No se puede extraer '" + newMethodName + "': valor de retorno ambiguo. "
                    + "Las siguientes variables necesitarían salir del método extraído: "
                    + candidateReturns.keySet()
                    + ". Igual que Eclipse, esta extracción se aborta; reduce o amplía la selección "
                    + "para que como mucho una variable quede pendiente de retorno.");
            return;
        }

        String returnVarName = null;
        Type returnVarType = null;
        boolean isDeclaredInside = false;

        if (!candidateReturns.isEmpty()) {
            returnVarName = candidateReturns.keySet().iterator().next();
            returnVarType = candidateReturns.get(returnVarName);
            isDeclaredInside = candidateIsDeclaredInside.get(returnVarName);
        }

        // --- ANÁLISIS DE JUMPS ---
        boolean hasEscapingContinue = false;
        boolean hasEscapingBreak = false;

        for (Statement stmt : statementsToExtract) {
            if (stmt.findAll(ContinueStmt.class).stream().anyMatch(c -> isEscapingContinue(c, statementsToExtract))) {
                hasEscapingContinue = true;
            }
            if (stmt.findAll(BreakStmt.class).stream().anyMatch(b -> isEscapingBreak(b, statementsToExtract))) {
                hasEscapingBreak = true;
            }
        }

        // --- CONSTRUCCIÓN DEL AST ---
        MethodDeclaration extractedMethod = new MethodDeclaration();
        extractedMethod.setName(newMethodName);
        extractedMethod.addModifier(Modifier.Keyword.PRIVATE);
        if (isStaticContext) {
            extractedMethod.addModifier(Modifier.Keyword.STATIC);
        }
        extractedMethod.setThrownExceptions(containerThrownExceptions);

        for (String param : requiredParams) {
            extractedMethod.addParameter(declaredBefore.get(param), param);
        }

        if (hasEscapingContinue || hasEscapingBreak) {
            extractedMethod.setType(StaticJavaParser.parseType("boolean"));
        } else {
            extractedMethod.setType(returnVarName != null ? returnVarType : new VoidType());
        }

        BlockStmt newBody = new BlockStmt();
        for (Statement stmt : statementsToExtract) {
            newBody.addStatement(stmt.clone());
        }

        // REESCRITURA DE JUMPS DENTRO DEL NUEVO MÉTODO
        if (hasEscapingContinue) {
            newBody.findAll(ContinueStmt.class).stream()
                    .filter(c -> !hasEnclosingLoop(c, newBody))
                    .forEach(c -> c.replace(new ReturnStmt(new BooleanLiteralExpr(true))));
        }

        if (hasEscapingBreak) {
            newBody.findAll(BreakStmt.class).stream()
                    .filter(b -> !hasEnclosingTarget(b, newBody))
                    .forEach(b -> b.replace(new ReturnStmt(new BooleanLiteralExpr(true))));
        }

        if (hasEscapingContinue || hasEscapingBreak) {
            newBody.addStatement(new ReturnStmt(new BooleanLiteralExpr(false)));
        } else if (returnVarName != null) {
            newBody.addStatement(new ReturnStmt(returnVarName));
        }

        extractedMethod.setBody(newBody);

        TypeDeclaration<?> parentClass = finalParentContainer.findAncestor(TypeDeclaration.class).orElseThrow();
        parentClass.addMember(extractedMethod);

        // --- CREACIÓN DEL REEMPLAZO ---
        MethodCallExpr call = new MethodCallExpr(null, newMethodName);
        requiredParams.forEach(call::addArgument);

        Statement replacementStmt;
        if (hasEscapingContinue || hasEscapingBreak) {
            Statement jumpStmt = hasEscapingContinue ? new ContinueStmt() : new BreakStmt();
            replacementStmt = new IfStmt(call, jumpStmt, null);
        } else if (returnVarName != null) {
            if (isDeclaredInside) {
                VariableDeclarationExpr vde = new VariableDeclarationExpr(
                        new VariableDeclarator(returnVarType, returnVarName, call)
                );
                replacementStmt = new ExpressionStmt(vde);
            } else {
                AssignExpr assign = new AssignExpr(new NameExpr(returnVarName), call, AssignExpr.Operator.ASSIGN);
                replacementStmt = new ExpressionStmt(assign);
            }
        } else {
            replacementStmt = new ExpressionStmt(call);
        }

        replacementStmt.setRange(targetRange);

        // --- INYECCIÓN SEGURA EN EL AST ---
        if (parentNode instanceof BlockStmt) {
            BlockStmt parentBlock = (BlockStmt) parentNode;
            int index = -1;
            NodeList<Statement> statements = parentBlock.getStatements();
            for (int i = 0; i < statements.size(); i++) {
                if (statements.get(i) == firstStmt) {
                    index = i;
                    break;
                }
            }

            if (index != -1) {
                parentBlock.addStatement(index, replacementStmt);
                for (Statement stmt : statementsToExtract) {
                    stmt.remove();
                }
            }
        } else if (parentNode instanceof MethodDeclaration && firstStmt instanceof BlockStmt) {
            BlockStmt newBodyParent = new BlockStmt();
            newBodyParent.addStatement(replacementStmt);
            ((MethodDeclaration) parentNode).setBody(newBodyParent);
        } else if (parentNode instanceof ConstructorDeclaration && firstStmt instanceof BlockStmt) {
            BlockStmt newBodyParent = new BlockStmt();
            newBodyParent.addStatement(replacementStmt);
            ((ConstructorDeclaration) parentNode).setBody(newBodyParent);
        } else if (parentNode instanceof InitializerDeclaration && firstStmt instanceof BlockStmt) {
            BlockStmt newBodyParent = new BlockStmt();
            newBodyParent.addStatement(replacementStmt);
            ((InitializerDeclaration) parentNode).setBody(newBodyParent);
        } else {
            firstStmt.replace(replacementStmt);
            for (int i = 1; i < statementsToExtract.size(); i++) {
                statementsToExtract.get(i).remove();
            }
        }
    }

    // --- NUEVO MOTOR DE RESOLUCIÓN DE RELACIONES ESTRUCTURALES ---

    private static boolean isNodeBefore(Node a, Node b) {
        if (a == b) return false;
        List<Node> pathA = getPathFromRoot(a);
        List<Node> pathB = getPathFromRoot(b);

        int minLen = Math.min(pathA.size(), pathB.size());
        Node commonAncestor = null;
        int diffIdx = -1;
        for (int i = 0; i < minLen; i++) {
            if (pathA.get(i) != pathB.get(i)) {
                diffIdx = i;
                break;
            }
            commonAncestor = pathA.get(i);
        }

        if (diffIdx == -1) {
            return pathA.size() < pathB.size();
        }

        Node siblingA = pathA.get(diffIdx);
        Node siblingB = pathB.get(diffIdx);

        List<Node> children = commonAncestor.getChildNodes();
        int idxA = children.indexOf(siblingA);
        int idxB = children.indexOf(siblingB);
        return idxA < idxB;
    }

    private static List<Node> getPathFromRoot(Node node) {
        List<Node> path = new ArrayList<>();
        Node current = node;
        while (current != null) {
            path.add(0, current);
            current = current.getParentNode().orElse(null);
        }
        return path;
    }

    private static boolean isAncestorOrSelf(Node ancestor, Node descendant) {
        Node current = descendant;
        while (current != null) {
            if (current == ancestor) {
                return true;
            }
            current = current.getParentNode().orElse(null);
        }
        return false;
    }

    private static boolean isInsideTarget(Node n, List<Statement> targetStmts) {
        for (Statement s : targetStmts) {
            if (isAncestorOrSelf(s, n)) {
                return true;
            }
        }
        return false;
    }

    private static Node getScopeDefiningNode(VariableDeclarator vd) {
        Node current = vd.getParentNode().orElse(null);
        while (current != null) {
            if (current instanceof ForStmt) {
                ForStmt fs = (ForStmt) current;
                if (fs.getInitialization().stream().anyMatch(init -> isAncestorOrSelf(init, vd))) {
                    return fs;
                }
            }
            if (current instanceof ForEachStmt) {
                ForEachStmt fes = (ForEachStmt) current;
                if (isAncestorOrSelf(fes.getVariable(), vd)) {
                    return fes;
                }
            }
            if (current instanceof TryStmt) {
                TryStmt ts = (TryStmt) current;
                if (ts.getResources().stream().anyMatch(res -> isAncestorOrSelf(res, vd))) {
                    return ts;
                }
            }
            if (current instanceof BlockStmt) {
                return current;
            }
            if (current instanceof MethodDeclaration || current instanceof ConstructorDeclaration || current instanceof InitializerDeclaration) {
                return current;
            }
            current = current.getParentNode().orElse(null);
        }
        return null;
    }

    private static boolean isVariableInScopeAtRange(VariableDeclarator vd, Statement firstStmt) {
        Node scopeNode = getScopeDefiningNode(vd);
        if (scopeNode == null) return false;
        return isAncestorOrSelf(scopeNode, firstStmt);
    }

    private static boolean isReferenceToNewDeclaration(NameExpr ne, List<Statement> targetStmts, BodyDeclaration<?> parentContainer) {
        String varName = ne.getNameAsString();
        Statement lastStmt = targetStmts.get(targetStmts.size() - 1);
        Optional<VariableDeclarator> newDecl = parentContainer.findAll(VariableDeclarator.class).stream()
                .filter(vd -> vd.getNameAsString().equals(varName))
                .filter(vd -> isNodeBefore(lastStmt, vd) && !isAncestorOrSelf(vd, lastStmt) && !isInsideTarget(vd, targetStmts))
                .filter(vd -> isNodeBefore(vd, ne) && !isAncestorOrSelf(ne, vd))
                .filter(vd -> {
                    Node scopeNode = getScopeDefiningNode(vd);
                    return scopeNode != null && isAncestorOrSelf(scopeNode, ne);
                })
                .findFirst();
        return newDecl.isPresent();
    }

    private static boolean canVariableEscape(VariableDeclarator vd, List<Statement> targetStmts) {
        Node scopeNode = getScopeDefiningNode(vd);
        if (scopeNode == null) return false;
        return !isInsideTarget(scopeNode, targetStmts);
    }

    // --- AUXILIARES ---

    private static boolean isEscapingContinue(ContinueStmt c, List<Statement> targetStmts) {
        Optional<Node> parent = c.getParentNode();
        while (parent.isPresent()) {
            Node p = parent.get();
            if (p instanceof ForStmt || p instanceof ForEachStmt || p instanceof WhileStmt || p instanceof DoStmt) {
                if (isInsideTarget(p, targetStmts)) {
                    return false;
                }
                return true;
            }
            parent = p.getParentNode();
        }
        return true;
    }

    private static boolean isEscapingBreak(BreakStmt b, List<Statement> targetStmts) {
        Optional<Node> parent = b.getParentNode();
        while (parent.isPresent()) {
            Node p = parent.get();
            if (p instanceof SwitchStmt || p instanceof ForStmt || p instanceof ForEachStmt || p instanceof WhileStmt || p instanceof DoStmt) {
                if (isInsideTarget(p, targetStmts)) {
                    return false;
                }
                return true;
            }
            parent = p.getParentNode();
        }
        return true;
    }

    private static boolean hasEnclosingLoop(Node node, Node limit) {
        Node parent = node.getParentNode().orElse(null);
        while (parent != null && parent != limit) {
            if (parent instanceof ForStmt || parent instanceof ForEachStmt || parent instanceof WhileStmt || parent instanceof DoStmt) {
                return true;
            }
            parent = parent.getParentNode().orElse(null);
        }
        return false;
    }

    private static boolean hasEnclosingTarget(Node node, Node limit) {
        Node parent = node.getParentNode().orElse(null);
        while (parent != null && parent != limit) {
            if (parent instanceof SwitchStmt || parent instanceof ForStmt || parent instanceof ForEachStmt || parent instanceof WhileStmt || parent instanceof DoStmt) {
                return true;
            }
            parent = parent.getParentNode().orElse(null);
        }
        return false;
    }

    private static String getContainerName(Statement stmt) {
        Optional<MethodDeclaration> m = stmt.findAncestor(MethodDeclaration.class);
        if (m.isPresent()) return m.get().getNameAsString();

        Optional<ConstructorDeclaration> c = stmt.findAncestor(ConstructorDeclaration.class);
        if (c.isPresent()) return c.get().getNameAsString();

        Optional<InitializerDeclaration> i = stmt.findAncestor(InitializerDeclaration.class);
        if (i.isPresent()) return i.get().isStatic() ? "static_init" : "instance_init";

        return "metodoDesconocido";
    }

    private static Position getPositionAtOffset(String content, int offset) {
        int line = 1, column = 1;
        for (int i = 0; i < offset && i < content.length(); i++) {
            if (content.charAt(i) == '\n') { line++; column = 1; }
            else { column++; }
        }
        return Position.pos(line, column);
    }

    private static boolean isWithin(Range nodeRange, Range targetRange) {
        if (nodeRange == null || targetRange == null) return false;
        return (nodeRange.begin.isAfter(targetRange.begin) || nodeRange.begin.equals(targetRange.begin)) &&
               (nodeRange.end.isBefore(targetRange.end) || nodeRange.end.equals(targetRange.end));
    }

    private static boolean hasAncestorInRange(Statement stmt, Range targetRange) {
        Optional<Node> parent = stmt.getParentNode();
        while (parent.isPresent()) {
            Node p = parent.get();
            if (p instanceof Statement && p.getRange().isPresent() && isWithin(p.getRange().get(), targetRange)) {
                return true;
            }
            parent = p.getParentNode();
        }
        return false;
    }
}