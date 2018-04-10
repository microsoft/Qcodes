echo "Executing the notebooks..."
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=600 --execute "examples\*.ipynb"
echo "Cleaning up the generated output..."
rm "examples\*nbconvert.ipynb"
