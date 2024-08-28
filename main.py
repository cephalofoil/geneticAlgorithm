import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab

# Parameter
pop_size = 10
gene_length = 8
mutation_rate = 0.05
max_generations = 100
cell_size = 50
diagram_width = 900  # Breiter für mehrere Diagramme
window_width = gene_length * cell_size + 100 + diagram_width
window_height = pop_size * cell_size + 100
fps = 1  # Frames per second, to control the speed of generation updates


# Initialisierung der Population mit Zufallswerten (0 oder 1)
def initialize_population():
    return np.random.randint(2, size=(pop_size, gene_length))


# Fitnessfunktion: Anzahl der 1en in einem Individuum
def fitness(individual):
    return np.sum(individual)


# Durchschnittliche Fitness der Population
def average_fitness(population):
    fitness_scores = np.array([fitness(ind) for ind in population])
    return np.mean(fitness_scores)


# Maximale Fitness der Population
def max_fitness(population):
    fitness_scores = np.array([fitness(ind) for ind in population])
    return np.max(fitness_scores)


# Minimale Fitness der Population
def min_fitness(population):
    fitness_scores = np.array([fitness(ind) for ind in population])
    return np.min(fitness_scores)


# Genetische Vielfalt der Population (Anzahl unterschiedlicher Genotypen)
def diversity(population):
    unique_genotypes = {tuple(ind) for ind in population}
    return len(unique_genotypes)


# Auswahl der Eltern basierend auf Fitness
def select_parents(population, fitness_scores):
    selected_parents = []
    for i, score in enumerate(fitness_scores):
        if random.random() < (score / 25.0):
            selected_parents.append(population[i])
    return selected_parents


# Einfache Crossover-Funktion (ein Punkt nach 4 Genen)
def crossover(parent1, parent2):
    crossover_point = gene_length // 2
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


# Mutation von Genen mit einer bestimmten Wahrscheinlichkeit
def mutate(child):
    for i in range(gene_length):
        if random.random() < mutation_rate:
            child[i] = 1 if child[i] == 0 else 0
    return child


# Evolutionäre Schritte (Selektion, Crossover, Mutation)
def evolve_population(population):
    fitness_scores = np.array([fitness(ind) for ind in population])
    parents = select_parents(population, fitness_scores)

    children = []
    while len(parents) >= 2:
        parent1 = parents.pop(random.randint(0, len(parents) - 1))
        parent2 = parents.pop(random.randint(0, len(parents) - 1))
        child = crossover(parent1, parent2)
        child = mutate(child)
        children.append(child)

    # Sortiere die Population nach Fitness (aufsteigend)
    sorted_population_indices = np.argsort(fitness_scores)

    # Ersetze die Individuen mit der geringsten Fitness durch die neuen Kinder
    num_children = len(children)
    for i in range(num_children):
        population[sorted_population_indices[i]] = children[i]

    return population


# Funktion, um die Population im Fenster darzustellen
def draw_population(screen, population, generation):
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)

    # Zeichne die Population
    for i in range(pop_size):
        for j in range(gene_length):
            # Rechtecke für die Gene (Schwarz für 1, Weiß für 0)
            color = (0, 0, 0) if population[i][j] == 1 else (255, 255, 255)
            pygame.draw.rect(screen, color, pygame.Rect(j * cell_size + 100, i * cell_size + 50, cell_size, cell_size))
            pygame.draw.rect(screen, (0, 0, 0),
                             pygame.Rect(j * cell_size + 100, i * cell_size + 50, cell_size, cell_size), 1)

    # Achsenbeschriftungen für Gene und Population
    screen.blit(small_font.render('Gene', True, (0, 0, 0)), (gene_length * cell_size // 2 + 100, 5))
    for j in range(gene_length):
        gene_label = small_font.render(f'{j + 1}', True, (0, 0, 0))
        screen.blit(gene_label, (j * cell_size + 100 + cell_size // 3, 30))

    # Rotate the "Individuals" label by 90 degrees
    individuals_label = small_font.render('Individuals', True, (0, 0, 0))
    individuals_label_rotated = pygame.transform.rotate(individuals_label, 90)
    screen.blit(individuals_label_rotated, (10, pop_size * cell_size // 2 + 50))

    for i in range(pop_size):
        pop_label = small_font.render(f'{i + 1}', True, (0, 0, 0))
        screen.blit(pop_label, (60, i * cell_size + 50 + cell_size // 4))

    # Anzeige der Generation unten links
    generation_text = font.render(f'Generation: {generation}', True, (0, 0, 255))
    screen.blit(generation_text, (10, window_height - 40))


# Funktion, um ein Diagramm zu erstellen und in pygame darzustellen
def create_chart(data, title, xlabel, ylabel, color, ylim=None):
    fig = pylab.figure(figsize=[3, 2],  # Size of the figure
                       dpi=100,  # DPI (dots per inch)
                       )
    ax = fig.gca()
    ax.plot(data, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")

    plt.close(fig)  # Schließe das Diagramm, um Speicherlecks zu vermeiden

    return surf


# Hauptschleife für die Evolution
def genetic_algorithm():
    pygame.init()

    # Fenster erstellen
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Genetischer Algorithmus - Evolution der Population')
    clock = pygame.time.Clock()

    population = initialize_population()
    running = True
    generation = 0

    fitness_history = []
    max_fitness_history = []
    min_fitness_history = []
    diversity_history = []

    while running and generation < max_generations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Hintergrund weiß
        draw_population(screen, population, generation)

        avg_fitness = average_fitness(population)
        max_fit = max_fitness(population)
        min_fit = min_fitness(population)
        diversity_count = diversity(population)

        fitness_history.append(avg_fitness)
        max_fitness_history.append(max_fit)
        min_fitness_history.append(min_fit)
        diversity_history.append(diversity_count)

        # Zeichne die Diagramme und blende sie im Fenster ein
        fitness_chart_surf = create_chart(fitness_history, 'Average Fitness', 'Generation', 'Fitness', 'blue', ylim=(0, gene_length))
        screen.blit(fitness_chart_surf, (gene_length * cell_size + 110, 50))

        max_fitness_chart_surf = create_chart(max_fitness_history, 'Max Fitness', 'Generation', 'Fitness', 'green', ylim=(0, gene_length))
        screen.blit(max_fitness_chart_surf, (gene_length * cell_size + 420, 50))

        min_fitness_chart_surf = create_chart(min_fitness_history, 'Min Fitness', 'Generation', 'Fitness', 'red', ylim=(0, gene_length))
        screen.blit(min_fitness_chart_surf, (gene_length * cell_size + 110, 300))

        diversity_chart_surf = create_chart(diversity_history, 'Population Diversity', 'Generation', 'Diversity', 'purple')
        screen.blit(diversity_chart_surf, (gene_length * cell_size + 420, 300))

        population = evolve_population(population)
        generation += 1

        pygame.display.flip()
        clock.tick(fps)  # Wartezeit, um die Geschwindigkeit zu kontrollieren

    pygame.quit()


if __name__ == "__main__":
    genetic_algorithm()
