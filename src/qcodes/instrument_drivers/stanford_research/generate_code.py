from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("."),
    lstrip_blocks=True,
    trim_blocks=True,
    keep_trailing_newline=True,
)
template = env.get_template("SR86x.jinja")

output = template.render()

with open("SR86x.py", "w") as f:
    f.write(output)
