echo "Executing the notebooks..."
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --execute "examples\DataSet\*.ipynb"


