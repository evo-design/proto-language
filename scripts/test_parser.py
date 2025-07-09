import json
import sys
sys.path.append('.')
from api.parser import GPLParser


with open("scripts/toy.json") as f:
    data = json.load(f)

parser = GPLParser(data)
program = parser.parse()
sequence_history = program.run()
