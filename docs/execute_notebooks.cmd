echo "Executing the notebooks..."
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=600 --execute "examples\DataSet\*.ipynb"
echo "Cleaning up the generated output..."
rm "examples\DataSet\*nbconvert.ipynb"
