# File: gui.py
# File Created: Friday, 10th February 2023 2:48:41 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 10th April 2023 10:15:39 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Contains GUI Class for 2048 in python
# Instructions found at https://techvidvan.com/tutorials/python-2048-game-project-with-source-code/

import tkinter as tk
from collections import defaultdict
from .backend import BACKEND_2048



class GAME_2048(tk.Frame):
    """2048 Game"""
    
    
    SUCCESSFUL_MOVE=0
    INVALID_MOVE=1
    GAME_OVER=2
    
    ###############
    # Color Codes #
    ###############
    
    Color_grid = "#b8afa9"
    Color_EmptyCell = "#ffd5b5"
    Font_ScoreLabel = ("Verdana", 24)
    Font_Score = ("Helvetica", 48, "bold")
    Font_GameOver = ("Helvetica", 48, "bold")
    Font_Color_GameOver = "#ffffff"
    Winner_BG = "#ffcc00"
    Loser_BG = "#a39489"
 
    Color_Cells = {
        2: "#fcefe6",
        4: "#f2e8cb",
        8: "#f5b682",
        16: "#f29446",
        32: "#ff775c",
        64: "#e64c2e",
        128: "#ede291",
        256: "#fce130",
        512: "#ffdb4a",
        1024: "#f0b922",
        2048: "#fad74d",
        4096: "#90EE90",
        8192: "#301934",
    }
 
    Color_CellNumber = {
        2: "#695c57",
        4: "#695c57",
        8: "#ffffff",
        16: "#ffffff",
        32: "#ffffff",
        64: "#ffffff",
        128: "#ffffff",
        256: "#ffffff",
        512: "#ffffff",
        1024: "#ffffff",
        2048: "#ffffff",
        4096: "#ffffff",
        8192: "#ffffff",
    }
 
    Fonts_CellNumber = {
        2: ("Helvetica", 55, "bold"),
        4: ("Helvetica", 55, "bold"),
        8: ("Helvetica", 55, "bold"),
        16: ("Helvetica", 50, "bold"),
        32: ("Helvetica", 50, "bold"),
        64: ("Helvetica", 50, "bold"),
        128: ("Helvetica", 45, "bold"),
        256: ("Helvetica", 45, "bold"),
        512: ("Helvetica", 45, "bold"),
        1024: ("Helvetica", 40, "bold"),
        2048: ("Helvetica", 40, "bold"),
        4096: ("Helvetica", 40, "bold"),
        8192: ("Helvetica", 40, "bold")
    }    
    
    
    def __init__(self):
        tk.Frame.__init__(self)

        self._setup_colors()
        
        
        # grid setup
        self.grid()
        self.master.title("2048")
 
        self.grid_main = tk.Frame(
            self, bg=GAME_2048.Color_grid, bd=3, width=600, height=600
        )
        self.grid_main.grid(pady=(110,0))
        
        # game over frame
        self.game_over_frame = None
        self.label_score = None
        self.cells = []
        
        self.reset_game()
        
    def reset_game(self):
        if self.game_over_frame:
            self.game_over_frame.destroy()
            
        if self.label_score:
            self.label_score.destroy()
            
        if len(self.cells) != 0:
            for row in self.cells:
                for cell_data in row:
                    cell_data['frame'].destroy()
            
        # create gui
        self.GUI_maker()
        
        # backend, start game
        self.backend = BACKEND_2048()
        self.GUI_update()
        
    def run_interactive_game(self):
        # bind controls.
        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<Up>", self.up)
        self.master.bind("<Down>", self.down)
 
        self.mainloop()
        
    def _setup_colors(self):
        
        # setup color cells
        tmp = defaultdict(lambda: "#000000")
        for key, value in self.Color_Cells.items():
            tmp[key] = value
        self.Color_Cells = tmp
        # setup cell number color
        tmp = defaultdict(lambda: "#ffffff")
        for key, value in self.Color_CellNumber.items():
            tmp[key] = value
        self.Color_CellNumber = tmp
        # setup font
        tmp = defaultdict(lambda x: 60 - 5 * len(str(x)))
        for key, value in self.Fonts_CellNumber.items():
            tmp[key] = value
        self.Fonts_CellNumber = tmp
        
     
    def GUI_maker(self):
        """Constructs the GUI for 2048
        """
        #   make grid
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                frame_cells = tk.Frame(
                    self.grid_main,
                    bg=GAME_2048.Color_EmptyCell,
                    width=150,
                    height=150
                )
                frame_cells.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.grid_main, bg=GAME_2048.Color_EmptyCell)
                cell_data = {"frame":frame_cells, "number": cell_number}
 
                cell_number.grid(row=i, column=j)
                row.append(cell_data)
            self.cells.append(row)

    
        # create score header
        frame_score = tk.Frame(self)
        frame_score.place(relx=0.5, y=60, anchor="center")
        tk.Label(
            frame_score,
            text="Score",
            font=GAME_2048.Font_ScoreLabel
        ).grid(row=0)
        self.label_score = tk.Label(frame_score, text="0", font= GAME_2048.Font_Score)
        self.label_score.grid(row=1)
        
        

    def left(self, event):
        """Left op"""
        stacked = self.backend.stack_left()
        if not self.backend.compress_left() and not stacked:
            return self.INVALID_MOVE
        self.backend.stack_left()
        # spawn a new tile
        self.backend.spawn_tile()
        # update GUI
        self.GUI_update()
        # check game over
        if self.backend.check_game_over():
            self.game_over()
            return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
            
    def right(self, event):
        """Right op"""
        # Convert to left-equiv
        self.backend.reverse()
        stacked = self.backend.stack_left()
        if not self.backend.compress_left() and not stacked:
            self.backend.reverse()
            return self.INVALID_MOVE
        self.backend.stack_left()
        self.backend.reverse()
        # spawn tile
        self.backend.spawn_tile()
        # update Gui
        self.GUI_update()
        # check game over
        if self.backend.check_game_over():
            self.game_over()
            return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
        
    def up(self, event):
        """Up op"""
        # convert to left-equiv
        self.backend.transpose()
        stacked = self.backend.stack_left()
        if not self.backend.compress_left() and not stacked:
            self.backend.transpose()
            return self.INVALID_MOVE
        self.backend.stack_left()
        self.backend.transpose()
        # spawn tile
        self.backend.spawn_tile()
        # update GUI
        self.GUI_update()
        # check Game over
        if self.backend.check_game_over():
            self.game_over() 
            return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
            
    def down(self, event):
        """Down op"""
        # convert to left-equivalent
        self.backend.transpose()
        self.backend.reverse()
        stacked = self.backend.stack_left()
        if not self.backend.compress_left() and not stacked:
            self.backend.reverse()
            self.backend.transpose()
            return self.INVALID_MOVE
        self.backend.stack_left()
        self.backend.reverse()
        self.backend.transpose()
        
        # spawn tile
        self.backend.spawn_tile()
        # update gui
        self.GUI_update()
        # check game over
        if self.backend.check_game_over(): 
            self.game_over() 
            return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
    
    
    def game_over(self):
        """Game over GUI"""
        self.game_over_frame = tk.Frame(self.grid_main, borderwidth=2)
        self.game_over_frame.place(relx=0.5, rely= 0.5, anchor="center")
        tk.Label(
            self.game_over_frame,
            text = "GAME OVER!!",
            bg=GAME_2048.Loser_BG,
            fg=GAME_2048.Font_Color_GameOver,
            font=GAME_2048.Font_GameOver
        ).pack()

        self.GUI_update()
    
    def GUI_update(self):
        """Updates the GUI
        """
        state = self.backend.get_state()
        for i in range(4):
            for j in range(4):
                cell_value = state[i][j]
                if cell_value == 0:
                    self.cells[i][j]["frame"].configure(bg=GAME_2048.Color_EmptyCell)
                    self.cells[i][j]["number"].configure(bg=GAME_2048.Color_EmptyCell, text="")
                else:
                    self.cells[i][j]["frame"].configure(bg=GAME_2048.Color_Cells[cell_value])
                    self.cells[i][j]["number"].configure(
                        bg=GAME_2048.Color_Cells[cell_value],
                        fg=GAME_2048.Color_CellNumber[cell_value],
                        font=GAME_2048.Fonts_CellNumber[cell_value],
                        text=str(cell_value)
                    )
        self.label_score.configure(text=self.backend.get_score())
        self.update_idletasks()