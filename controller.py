from star_model import Model1


class Controller:
    def __init__(self, view):
        self.view = view
        self.model = Model1()     # using Strategy Model1 for now

    def train_model_clicked(self):
        try:
            self.model.trainmodel()
            print("Model training complete.")
        except Exception as e:
            print("Error training model:", e)

    def view_graph_clicked(self):
        print("Graph view requested. (Not implemented yet)")
