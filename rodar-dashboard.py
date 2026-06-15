from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from pathlib import Path

pkl_dir = Path.cwd() / "pkls"
caminho_do_arquivo = pkl_dir / "explainer.pkl"

explainer_salvo = ClassifierExplainer.from_file(caminho_do_arquivo)

db = ExplainerDashboard(explainer_salvo)
db.run()