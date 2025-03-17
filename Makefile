venv-dev:
	conda env create --name forestid_venv_dev --file environment.yml 
	conda run -n forestid_venv_dev pip install -e .[dev]

update-venv-dev:
	conda run -n forestid_venv_dev pip install -e .[dev]

venv:
	conda env create --name forestid_venv --file environment.yml
	conda run -n forestid_venv pip install -e .

update-venv:
	conda run -n forestid_venv pip install -e .