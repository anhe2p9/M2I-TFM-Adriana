import os
import re
from openai import OpenAI
from datetime import datetime

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.1"   # o "gpt-5.1"
INPUT_FOLDER = "codigos_original_y_refactorizado"

client = OpenAI(api_key=API_KEY)


def find_java_pairs(folder):
    """Returns pairs list (original_path, refactored_path)"""
    originals = {}
    refactors = {}

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.endswith(".java"):
                continue
            if f.startswith("original_"):
                key = f[len("original_"):]
                originals[key] = os.path.join(root, f)
            elif f.startswith("refactored_"):
                key = f[len("refactored_"):]
                refactors[key] = os.path.join(root, f)

    pairs = []
    for key in originals:
        if key in refactors:
            pairs.append((originals[key], refactors[key]))

    return pairs


def extract_extraction_names(java_content):
    """
    Returns just the method names with _extractionX.
    It does not returns the content, just the names.
    """
    pattern = r"\b(\w+_extraction\d+)\b"
    names = set(re.findall(pattern, java_content))
    return sorted(names)


def send_to_chatgpt(original_code, refactored_code, extraction_name, used_names):
    """
    Sendes prompt + content from both JAva files as text.
    Considers the method manes used to not repeat them.
    """
    # Create string with used names
    used_str = ""
    if used_names:
        used_str = f" Do not repeat the following method names: {', '.join(used_names)}."

    prompt = (
        f"Given the initial Java code: {original_code} and"
        f" its refactored version: {refactored_code}, "
        f"Can you give a name for the extracted method {extraction_name}? "
        f"Provide the method name without parentheses. "
        f"Avoid the extract operation bias for the method name prediction."
        f" Use Java notation for the method name (for example, do not use _)."
        f"Do not use any of the following names: {used_str}."
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ]
    )

    return response.output_text.strip()


def main():
    log = ""  # Accumulates all the log
    pairs = find_java_pairs(INPUT_FOLDER)

    if not pairs:
        print("Pairs of original/refactored version not found.")
        return

    for original_path, refactored_path in pairs:
        header = f"\nProcessing:\n - {original_path}\n - {refactored_path}\n"
        print(header)
        log += header

        # Read codes
        with open(original_path, "r", encoding="utf-8") as f:
            original_code = f.read()
        with open(refactored_path, "r", encoding="utf-8") as f:
            refactored_code = f.read()

        extraction_names = extract_extraction_names(refactored_code)

        if not extraction_names:
            msg = "No extraction found *_extractionX.\n"
            print(msg)
            log += msg
            continue

        used_names = []

        # Copy that we will be modifying locally
        modified_code = refactored_code

        for extraction in extraction_names:
            msg = f"  → Sending extraction: {extraction}\n"
            print(msg, end="")
            log += msg

            # Call to ChatGPT always with refactored_ version
            new_name = send_to_chatgpt(original_code, refactored_code, extraction, used_names)
            used_names.append(new_name)

            # Replace the method mane in the modified code
            pattern = rf"\b{re.escape(extraction)}\b"
            modified_code = re.sub(pattern, new_name, modified_code)

            print(f"    ✓ New name: {new_name}\n")
            log += f"    ✓ New name: {new_name}\n"

        # Save updated code in a new file
        new_file_path = os.path.join(
            os.path.dirname(refactored_path),
            f"refactored_renamed_{os.path.basename(refactored_path)}"
        )
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(modified_code)

        print(f"    ✓ Updated refactored file saved in {new_file_path}\n")

    # Save complete log in an unic file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(INPUT_FOLDER, f"complete_log_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log)

    print(f"\n✅ Complete log saved in {log_file}")


if __name__ == "__main__":
    main()
