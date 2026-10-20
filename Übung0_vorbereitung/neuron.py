"""Das besipiel zeigt die implemnetierung eines einfachen Neurons in Python.
wie es in den vorlesungsunterlagen von hern prof dr liebscher beschrieben ist.
"""
import numpy as np

class neuron : 
   
    input = {}
    weight = {}
    output = 0.0

    # konstrucktor
    def __init__(self, input):
        # Initialisiert die Gewichte zufällig
        self.input = input
        self.weight = np.random.rand(len(input))  # Zufällige Gewichte für jede Eing
                              
    # Berechnet die gewichtete Summe der Eingaben
    # sum (x*w)
    def sum (self, inputs):
        # Berechnet die gewichtete Summe der Eingaben
        return np.sum(inputs * self.weight)
        
  
    def aktivierungsfunktion(self, x):
        # Sigmoid-Aktivierungsfunktion
        return 1 / (1 + np.exp(-x))

    def forward(self):
        # Berechnet die gewichtete Summe der Eingaben
        weighted_sum = self.sum(self.input)
        # Wendet die Aktivierungsfunktion an
        self.output = self.aktivierungsfunktion(weighted_sum)
        return self.output

    def train(self, target_output, learning_rate=0.1):
        # Berechnet den Fehler
        error = target_output - self.output
        # Aktualisiert die Gewichte basierend auf dem Fehler und der Lernrate
        self.weight += learning_rate * error * self.input

    
# SIMPLER TEST DES NEURONS
from Übung1.neuron import neuron

if __name__ == "__main__":
    # Beispiel für die Verwendung des Neurons
    inputs = np.array([0.5, 0.2, 0.8])  # Beispielhafte Eingaben

    my_neuron = neuron(inputs)

    # Ausgabe vor dem Training 
    print("Vorher:", my_neuron.forward())

    # gewünschter Output: eine einzelne Zahl 
    target_output = 1.0

    # training
    for _ in range(1000):
        my_neuron.train(target_output)


    print("Nachher:", my_neuron.forward())
   
    



 