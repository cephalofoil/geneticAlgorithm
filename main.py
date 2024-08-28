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
diagram_width = 300
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
    screen.fill((255, 255, 255))  # Hintergrund weiß

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

    pygame.display.flip()


# Funktion, um das Fitness-Diagramm zu aktualisieren und anzuzeigen
def draw_fitness_chart(fitness_history):
    fig = pylab.figure(figsize=[3, 5],  # Size of the figure
                       dpi=100,  # DPI (dots per inch)
                       )
    ax = fig.gca()
    ax.plot(fitness_history, color='blue')
    ax.set_title('Average Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_ylim(0, gene_length)

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")

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

    while running and generation < max_generations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_population(screen, population, generation)

        avg_fitness = average_fitness(population)
        fitness_history.append(avg_fitness)

        # Zeichne das Fitness-Diagramm und blende es auf der rechten Seite ein
        fitness_chart_surf = draw_fitness_chart(fitness_history)
        screen.blit(fitness_chart_surf, (gene_length * cell_size + 110, 50))

        population = evolve_population(population)
        generation += 1

        pygame.display.flip()
        clock.tick(fps)  # Wartezeit, um die Geschwindigkeit zu kontrollieren

    pygame.quit()


if __name__ == "__main__":
    genetic_algorithm()