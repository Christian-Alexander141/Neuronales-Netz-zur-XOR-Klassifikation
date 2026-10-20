""""Das neuron bearbeitet das AND gatter"""

import numpy as np

class neuron : 
   
    input = {}
    weight = {}
    output = 0.0
    bias = 0.0

    # konstrucktor
    def __init__(self, input):
        # Übergebene Eingaben speichern
        self.input = np.asarray(input, dtype=float)

        # Zufällige Gewichte für jede Eingabe erzeugen
        self.weight = np.random.uniform(
            -0.1, 0.1, size=len(self.input)
        )

        # Bias initialisieren
        self.bias = 0.0

        # Output initialisieren
        self.output = 0.0
                              
    # Berechnet die gewichtete Summe der Eingaben
    # sum (x*w)
    def sum(self, inputs):
        return np.sum(inputs * self.weight) + self.bias
        
  
    def aktivierungsfunktion(self, x):
        # Sigmoid-Aktivierungsfunktion
        return 1 / (1 + np.exp(-x))

    def forward(self):
        # Berechnet die gewichtete Summe der Eingaben
        weighted_sum = self.sum(self.input)
        # Wendet die Aktivierungsfunktion an
        self.output = self.aktivierungsfunktion(weighted_sum)
        return self.output

   # Ein Trainingsschritt
    def train(self, target_output, learning_rate=0.1):

        # 1. Aktuellen Output berechnen
        output = self.forward()

        # 2. Fehler berechnen
        error = target_output - output

        # 3. Lernregel für die Sigmoid-Funktion
        delta = error * output * (1 - output)

        # 4. Gewichte anpassen
        self.weight += learning_rate * delta * self.input

        # 5. Bias anpassen
        self.bias += learning_rate * delta


# Trainingsbeispiele
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Zielwerte
y = np.array([0, 0, 0, 1], dtype=float)

np.random.seed(42)

# Neuron mit zwei Eingaben erstellen
my_neuron = neuron(X[0])

# Training
for epoche in range(10000):
    for inputs, target in zip(X, y):

        # Jeweils einen Eingabevektor verwenden
        my_neuron.input = inputs

        # Einen einzelnen Trainingsschritt durchführen
        my_neuron.train(target, learning_rate=0.5)

# Ergebnisse überprüfen
for inputs, target in zip(X, y):
    my_neuron.input = inputs
    output = my_neuron.forward()

    prediction = int(output >= 0.5)

    print(
        f"Eingabe: {inputs.astype(int)}, "
        f"Output: {output:.3f}, "
        f"Vorhersage: {prediction}, "
        f"Ziel: {int(target)}"
    )