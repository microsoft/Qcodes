echo "Executing the notebooks..."
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=600 --execute "examples\DataSet\*.ipynb"


