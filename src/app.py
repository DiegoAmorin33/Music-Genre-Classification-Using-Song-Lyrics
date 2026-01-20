from utils import db_connect
engine = db_connect()

bad_lines = []

def bad_line_handler(line):
    bad_lines.append(line)
    return None   # None = skip la línea

df = pd.read_csv(
    "lyrics,label.txt",
    sep=",",
    engine="python",
    on_bad_lines=bad_line_handler
)

print("Cantidad de líneas malas:", len(bad_lines))
print("Ejemplo de línea mala:")
print(bad_lines[0])
