"""Main module."""

import numpy as np

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay=0.5, alpha=1, beta=1):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        # Initialize pheromone trails
        pheromones = np.ones(self.distances.shape) / len(self.distances)

        # Run iterations
        for i in range(self.n_iterations):
            # Create ants and run tours
            ants = [Ant(pheromones, self.alpha, self.beta) for _ in range(self.n_ants)]
            tours = [ant.run_tour(self.distances) for ant in ants]

            # Update pheromone trails
            for i, tour in enumerate(tours):
                for j in range(len(tour) - 1):
                    a, b = tour[j], tour[j + 1]
                    pheromones[a][b] += 1 / self.distances[a][b]
                    pheromones[b][a] += 1 / self.distances[b][a]

            # Decay pheromones
            pheromones *= self.decay

        # Find best tour
        best_tour = min(tours, key=lambda x: self.get_tour_distance(x))
        return best_tour

    def get_tour_distance(self, tour):
        # Calculate the total distance of a tour
        distance = 0
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            distance += self.distances[a][b]
        return distance


class Ant:
    def __init__(self, pheromones, alpha, beta):
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta

    def run_tour(self, distances):
        # Create a new tour and add the starting node
        tour = [0]

        # Visit each node in the tour
        while len(tour) < len(distances):
            current_node = tour[-1]

            # Calculate the probabilities of moving to each neighbor
            pheromone = self.pheromones[current_node]
            distance = distances[current_node]
            probabilities = np.power(pheromone, self.alpha) * np.power(1 / distance, self.beta)
            probabilities /= probabilities.sum()

            # Choose the next node based on the probabilities
            next_node = np.random.choice(len(distances), p=probabilities)
            tour.append(next_node)

        # Add the starting node to the end of the tour
        tour.append(0)
        return tour

if __name__ == '__main__':
    distances = np.array([[0, 2, 2, 5], [2, 0, 4, 1], [2, 4, 0, 3], [5, 1, 3, 0]])
    colony = AntColony(distances=distances, n_ants=100, n_iterations=10)
    colony.run()
