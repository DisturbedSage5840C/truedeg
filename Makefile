VENV = .venv/bin/python

.PHONY: smoke data figures deck model app clean

smoke:
	$(VENV) -m src.build_dataset --smoke

data:
	$(VENV) -m src.build_dataset

figures:
	$(VENV) -m src.experiments

deck:
	$(VENV) -m src.regression
	$(VENV) -m src.deck_figures

model:
	$(VENV) -m src.model

app:
	.venv/bin/streamlit run app.py

clean:
	rm -f figures/*.png figures/*.csv
	# never removes data/cache/
