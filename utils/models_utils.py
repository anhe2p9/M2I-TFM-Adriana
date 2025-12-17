import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

def process_data(s_filename, n_filename, c_filename, offsets_filename) -> dict:

    data = dp.DataPortal()

    empty_file = []
    missing_file = []
    if s_filename:
        with open(str(s_filename), 'r', encoding='utf-8') as f:
            if sum(1 for _ in f) > 1:  # at least there must be one nested sequence
                data.load(filename=str(s_filename), index="S", param=("loc", "nmcc", "params"))
            else:
                empty_file.append("sequences")
    else:
        missing_file.append("sequences")

    if n_filename:
        with open(str(n_filename), 'r', encoding='utf-8') as f:
            if sum(1 for _ in f) > 1:
                data.load(filename=str(n_filename), index="N", param="ccr")
            else:
                empty_file.append("nested")
    else:
        missing_file.append("nested")

    if c_filename:
        with open(str(c_filename), 'r', encoding='utf-8') as f:
            if sum(1 for _ in f) > 1:
                data.load(filename=str(c_filename), index="C", param=())
            else:
                empty_file.append("conflict")
    else:
        missing_file.append("conflict")

    if offsets_filename:
        with open(str(offsets_filename), 'r', encoding='utf-8') as f:
            if not f.read():
                empty_file.append("offsets")
    else:
        missing_file.append("offsets")

    total_data = {"missingFiles": missing_file, "emptyFiles": empty_file, "data": data, "offsets": offsets_filename}
    # print(f"DATA: {total_data}")
    return total_data