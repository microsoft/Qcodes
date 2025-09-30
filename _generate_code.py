from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("src/qcodes/instrument_drivers/stanford_research"),
    lstrip_blocks=True,
    trim_blocks=True,
    keep_trailing_newline=True,
)
template = env.get_template("SR86x.jinja")

output = template.render()

with open("src/qcodes/instrument_drivers/stanford_research/SR86x.py", "w") as f:
    f.write(output)
