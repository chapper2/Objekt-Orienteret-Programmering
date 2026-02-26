import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, Button
from collections import defaultdict

class ForestFireSimulation:
    """
    Forest fire simulation converted from NetLogo.
    Simulates fire spreading through a forest with density-based tree distribution.
    """
    
    def __init__(self, grid_size=50, density=60):
        """
        Initialize the forest fire simulation.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            density: Percentage of patches that will be trees (0-100)
        """
        self.grid_size = grid_size
        self.density = density
        
        # Grid states
        self.EMPTY = 0
        self.TREE = 1
        self.FIRE = 2
        self.EMBER_1 = 3
        self.EMBER_2 = 4
        self.EMBER_3 = 5
        self.EMBER_4 = 6
        self.BURNED = 7
        
        # Statistics
        self.initial_trees = 0
        self.burned_trees = 0
        self.ticks = 0
        
        # Grid
        self.grid = None
        
        # For visualization
        self.fig = None
        self.ax = None
        self.im = None
        
    def setup(self):
        """Initialize the simulation - create trees and start the fire."""
        # Create empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Plant trees based on density
        random_values = np.random.random((self.grid_size, self.grid_size)) * 100
        self.grid[random_values < self.density] = self.TREE
        
        # Ignite the left edge
        for y in range(self.grid_size):
            if self.grid[y, 0] == self.TREE or self.grid[y, 0] == self.EMPTY:
                self.grid[y, 0] = self.FIRE
                if self.grid[y, 0] == self.TREE:
                    self.burned_trees += 1
        
        # Count initial trees (excluding the burning edge)
        self.initial_trees = np.sum(self.grid == self.TREE) + self.burned_trees
        self.ticks = 0
        
    def get_neighbors4(self, y, x):
        """Get the 4 neighboring cells (up, down, left, right)."""
        neighbors = []
        # Up
        if y > 0:
            neighbors.append((y - 1, x))
        # Down
        if y < self.grid_size - 1:
            neighbors.append((y + 1, x))
        # Left
        if x > 0:
            neighbors.append((y, x - 1))
        # Right
        if x < self.grid_size - 1:
            neighbors.append((y, x + 1))
        return neighbors
    
    def ignite(self, y, x):
        """Ignite a tree at position (y, x)."""
        if self.grid[y, x] == self.TREE:
            self.grid[y, x] = self.FIRE
            self.burned_trees += 1
    
    def go(self):
        """Run one step of the simulation."""
        # Check if there are any fires or embers
        if not np.any((self.grid == self.FIRE) | 
                      (self.grid == self.EMBER_1) |
                      (self.grid == self.EMBER_2) |
                      (self.grid == self.EMBER_3) |
                      (self.grid == self.EMBER_4)):
            return False  # Simulation finished
        
        # Find all fire positions
        fire_positions = np.argwhere(self.grid == self.FIRE)
        
        # New grid state
        new_grid = self.grid.copy()
        
        # Spread fire to neighboring trees
        for y, x in fire_positions:
            neighbors = self.get_neighbors4(y, x)
            for ny, nx in neighbors:
                if self.grid[ny, nx] == self.TREE:
                    new_grid[ny, nx] = self.FIRE
                    self.burned_trees += 1
            # Fire becomes ember
            new_grid[y, x] = self.EMBER_1
        
        # Fade embers
        new_grid[self.grid == self.EMBER_1] = self.EMBER_2
        new_grid[self.grid == self.EMBER_2] = self.EMBER_3
        new_grid[self.grid == self.EMBER_3] = self.EMBER_4
        new_grid[self.grid == self.EMBER_4] = self.BURNED
        
        self.grid = new_grid
        self.ticks += 1
        return True  # Continue simulation
    
    def get_color_map(self):
        """Create a color map for visualization."""
        # Colors: Empty (white), Tree (green), Fire (red), Embers (fading red), Burned (black)
        colors = [
            '#FFFFFF',  # EMPTY - white
            '#228B22',  # TREE - forest green
            '#FF0000',  # FIRE - bright red
            '#E60000',  # EMBER_1 - red
            '#CC0000',  # EMBER_2 - darker red
            '#990000',  # EMBER_3 - even darker
            '#660000',  # EMBER_4 - very dark red
            '#1A1A1A',  # BURNED - near black
        ]
        return ListedColormap(colors)
    
    def visualize_static(self):
        """Display the current state of the grid."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap=self.get_color_map(), vmin=0, vmax=7)
        plt.title(f'Forest Fire - Tick {self.ticks}\n'
                  f'Burned: {self.burned_trees}/{self.initial_trees} trees '
                  f'({self.burned_trees/max(self.initial_trees, 1)*100:.1f}%)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def run_simulation(self, interval=100, save_animation=False):
        """
        Run the full simulation with animation.
        
        Args:
            interval: Milliseconds between frames
            save_animation: If True, saves animation as MP4 (requires ffmpeg)
        """
        self.setup()
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.im = self.ax.imshow(self.grid, cmap=self.get_color_map(), 
                                  vmin=0, vmax=7, interpolation='nearest')
        self.ax.axis('off')
        
        title = self.ax.text(0.5, 1.05, '', transform=self.ax.transAxes,
                            ha='center', fontsize=14)
        
        def update(frame):
            """Update function for animation."""
            # Run one step
            continuing = self.go()
            
            if not continuing:
                ani.event_source.stop()
            
            # Update display
            self.im.set_array(self.grid)
            title.set_text(f'Forest Fire - Tick {self.ticks}\n'
                          f'Burned: {self.burned_trees}/{self.initial_trees} trees '
                          f'({self.burned_trees/max(self.initial_trees, 1)*100:.1f}%)')
            return [self.im, title]
        
        ani = animation.FuncAnimation(self.fig, update, frames=1000, 
                                     interval=interval, blit=True, repeat=False)
        
        if save_animation:
            ani.save('forest_fire.mp4', writer='ffmpeg', fps=10)
            print("Animation saved as 'forest_fire.mp4'")
        
        plt.tight_layout()
        plt.show()
    
    def run_without_animation(self, max_steps=1000, show_final=True):
        """
        Run simulation without animation (faster).
        
        Args:
            max_steps: Maximum number of steps to run
            show_final: If True, shows the final state
        """
        self.setup()
        
        step = 0
        while step < max_steps and self.go():
            step += 1
            if step % 10 == 0:
                print(f"Step {step}: {self.burned_trees}/{self.initial_trees} trees burned")
        
        print(f"\nSimulation finished after {self.ticks} ticks")
        print(f"Total burned: {self.burned_trees}/{self.initial_trees} trees "
              f"({self.burned_trees/max(self.initial_trees, 1)*100:.1f}%)")
        
        if show_final:
            self.visualize_static()


def run_interactive_simulation(grid_size=50):
    """
    Run simulation with interactive controls (slider for density and begin button).
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
    """
    # Create figure and axes
    fig = plt.figure(figsize=(12, 10))
    
    # Main plot area
    ax_main = plt.axes([0.1, 0.25, 0.8, 0.65])
    ax_main.axis('off')
    
    # Slider for density
    ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
    slider_density = Slider(ax_slider, 'Tree Density (%)', 0, 100, valinit=60, valstep=1)
    
    # Begin button
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
    button_begin = Button(ax_button, 'Begin Simulation', color='lightgreen', hovercolor='green')
    
    # Create simulation object
    sim = ForestFireSimulation(grid_size=grid_size, density=60)
    
    # Animation object (will be created when button is clicked)
    ani = None
    running = False
    
    # Initial setup
    sim.setup()
    im = ax_main.imshow(sim.grid, cmap=sim.get_color_map(), vmin=0, vmax=7, interpolation='nearest')
    title = ax_main.text(0.5, 1.05, f'Forest Fire - Ready to Start\nDensity: {int(slider_density.val)}%',
                         transform=ax_main.transAxes, ha='center', fontsize=14)
    
    def update_density(val):
        """Update the simulation when density slider changes."""
        nonlocal running
        if not running:  # Only allow changes when not running
            density = int(slider_density.val)
            sim.density = density
            sim.setup()
            im.set_array(sim.grid)
            title.set_text(f'Forest Fire - Ready to Start\nDensity: {density}%')
            fig.canvas.draw_idle()
    
    def begin_simulation(event):
        """Start the simulation when begin button is clicked."""
        nonlocal ani, running
        
        if running:  # Reset if already running
            if ani is not None:
                ani.event_source.stop()
        
        # Reset simulation with current density
        density = int(slider_density.val)
        sim.density = density
        sim.setup()
        running = True
        
        # Disable slider during simulation
        slider_density.set_active(False)
        button_begin.label.set_text('Reset')
        button_begin.color = 'lightcoral'
        
        def update(frame):
            """Update function for animation."""
            nonlocal running
            
            # Run one step
            continuing = sim.go()
            
            if not continuing:
                ani.event_source.stop()
                running = False
                slider_density.set_active(True)
                button_begin.label.set_text('Begin Simulation')
                button_begin.color = 'lightgreen'
            
            # Update display
            im.set_array(sim.grid)
            title.set_text(f'Forest Fire - Tick {sim.ticks}\n'
                          f'Burned: {sim.burned_trees}/{sim.initial_trees} trees '
                          f'({sim.burned_trees/max(sim.initial_trees, 1)*100:.1f}%)')
            return [im, title]
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=1000, 
                                     interval=50, blit=True, repeat=False)
        fig.canvas.draw_idle()
    
    # Connect slider and button to their callback functions
    slider_density.on_changed(update_density)
    button_begin.on_clicked(begin_simulation)
    
    plt.show()


def main():
    """Main function to run the forest fire simulation."""
    
    # Run interactive simulation with controls
    run_interactive_simulation(grid_size=50)
    
    # Alternative: Run with animation (no controls)
    # sim = ForestFireSimulation(grid_size=50, density=60)
    # sim.run_simulation(interval=50)
    
    # Alternative: Run without animation (faster)
    # sim = ForestFireSimulation(grid_size=50, density=60)
    # sim.run_without_animation(show_final=True)


if __name__ == "__main__":
    main()


# Copyright notice from original NetLogo code:
# Copyright 1997 Uri Wilensky.
# Converted to Python 2026
