import sys
sys.path.append('.')

from language.tools import predict_structure_esmfold


if __name__ == '__main__':
    esmfold_output = predict_structure_esmfold(sequences=sys.argv[1])

    esmfold_output.save_pdb("design.pdb")
