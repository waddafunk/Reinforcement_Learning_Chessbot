from random import choice

import chess
import gym
import numpy as np
import pygame
from gym import spaces

from constants import N_ACTIONS

# Constants for the chessboard
BOARD_SIZE = 1000
SQUARE_SIZE = BOARD_SIZE // 8
COLORS = {
    "white": (255, 255, 255),
    "black": (150, 150, 150),
    "highlight": (100, 200, 100),
}
SYMBOL_TO_PIECE_NAME = {
    "k": "king",
    "q": "queen",
    "r": "rook",
    "b": "bishop",
    "n": "knight",
    "p": "pawn",
}


class ChessEnv(gym.Env):
    def __init__(self):
        self.board = chess.Board()
        # Actions: 64 possible 'from' squares * 64 possible 'to' squares * 5 possible promotions
        self.action_space = spaces.Discrete(N_ACTIONS)
        # Observation: 8x8 board with integer values representing pieces
        self.observation_space = spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8)
        self.render_function = self.init_render

    def move_to_action(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion

        # Total number of possible moves without promotions
        non_promotion_moves = 64 * 63

        if promotion is None:
            # Non-promotion move
            action = from_square * 63 + to_square
            if to_square > from_square:
                action -= 1
            return action

        # Promotion move
        from_file = chess.square_file(from_square)
        to_file = chess.square_file(to_square)

        promotion_map = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3,
        }

        action = (
            non_promotion_moves
            + (from_file * 32)
            + (to_file * 4)
            + promotion_map[promotion]
        )
        return action

    def turn(self):
        return self.board.turn

    # Helper function to convert UCI string to Move object
    @staticmethod
    def uci_to_move(uci_string):
        return chess.Move.from_uci(uci_string)

    def sample(self):
        move = choice(list(self.board.legal_moves))
        return self.move_to_action(move)

    def init_graphics(self):
        pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
        pygame.display.set_caption("Chess Environment")

    def render_no_init(self, _="human"):
        self._draw_board()
        pygame.display.flip()

    def init_render(self, _="human"):
        self.init_graphics()
        self.render_no_init()
        self.render_function = self.render_no_init

    def render(self, mode="human"):
        self.render_function(mode)

    def _draw_board(self):
        screen = pygame.display.get_surface()
        screen.fill(COLORS["black"])

        for i in range(8):
            for j in range(8):
                color = COLORS["white"] if (i + j) % 2 == 0 else COLORS["black"]
                pygame.draw.rect(
                    screen,
                    color,
                    (i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )

        for square, piece in self.board.piece_map().items():
            self._draw_piece(piece, square)

    def _draw_piece(self, piece, square):
        color = "black"
        if piece.color:
            color = "white"

        screen = pygame.display.get_surface()
        image = pygame.image.load(
            "images/"
            + SYMBOL_TO_PIECE_NAME[piece.symbol().lower()]
            + "_"
            + color
            + ".png"
        )
        image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
        rank, file = divmod(square, 8)
        screen.blit(image, (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE))

    def reset(self, *, seed=None, options=None):
        if seed is not None or options is not None:
            print("WARNING: seed and option args not used")
        self.board.reset()
        return self._get_observation()

    def step(self, action):
        # Convert the action to a chess move
        from_square, to_square, promotion = self.action_to_move(action)
        move = chess.Move(from_square, to_square, promotion=promotion)

        # Check if the move is legal
        if move in self.board.legal_moves:
            # Perform the move on the chess board
            self.board.push(move)

            # Check if the game is over
            done = self.board.is_game_over()

            # Calculate the reward based on the game outcome
            if self.board.is_checkmate():
                reward = 100.0  # if self.board.turn == chess.WHITE else -1.0
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                reward = 0.0
            else:
                reward = 0.0

            # Get the new observation after the move
            observation = self._get_observation()

            return observation, reward, done, {}

        # Invalid move, return the current observation and a negative reward
        return self._get_observation(), -1.0, False, {}

    def action_to_move(self, action):
        # Total number of possible moves without promotions
        non_promotion_moves = 64 * 63

        if action < non_promotion_moves:
            # Non-promotion move
            from_square = action // 63
            to_square = action % 63
            if to_square >= from_square:
                to_square += 1
            return (chess.SQUARES[from_square], chess.SQUARES[to_square], None)

        # Promotion move
        action -= non_promotion_moves
        from_file = action // (8 * 4)
        to_file = (action % (8 * 4)) // 4
        promotion = action % 4

        promotion_map = {
            0: chess.QUEEN,
            1: chess.ROOK,
            2: chess.BISHOP,
            3: chess.KNIGHT,
        }

        # Determine the rank based on the color to play
        from_rank = 6 if self.board.turn == chess.WHITE else 1
        to_rank = 7 if self.board.turn == chess.WHITE else 0
        from_square = chess.square(from_file, from_rank)
        to_square = chess.square(to_file, to_rank)

        return (from_square, to_square, promotion_map[promotion])

    def _get_observation(self):
        # Convert the board state to a feature representation
        feature_matrix = np.zeros((8, 8), dtype=np.int8)

        for square, piece in self.board.piece_map().items():
            rank, file = divmod(square, 8)
            piece_type = piece.piece_type
            color = piece.color

            # Assign integer values based on piece type and color
            piece_value = piece_type
            if color == chess.BLACK:
                piece_value *= -1

            feature_matrix[rank, file] = piece_value

        return feature_matrix.flatten()

    def explain_action(self, action):
        if action is None:
            return "Invalid move"

        from_square, to_square, promotion = self.action_to_move(action)

        # Get the piece at the from_square
        piece = self.board.piece_at(from_square)

        # Convert the square indices to algebraic notation
        from_square_str = chess.SQUARE_NAMES[from_square]
        to_square_str = chess.SQUARE_NAMES[to_square]

        if piece is None:
            return f"Invalid move: {from_square_str}{to_square_str}"

        # Get the piece type and color
        piece_type = chess.PIECE_NAMES[piece.piece_type]
        color = "White" if piece.color == chess.WHITE else "Black"

        # Create the move explanation
        move_explanation = (
            f"{color} {piece_type} moves from {from_square_str} to {to_square_str}"
        )

        # Handle promotion
        if promotion is not None:
            promotion_piece = chess.PIECE_NAMES[promotion]
            move_explanation += f" and promotes to {promotion_piece}"

        return move_explanation

    def notation_to_action(self, notation):
        try:
            # Parse the notation using python-chess
            move = chess.Move.from_uci(notation)

            # Get the from_square and to_square
            from_square = move.from_square
            to_square = move.to_square

            # Get the promotion piece type (if applicable)
            promotion = move.promotion

            # Convert the promotion piece type to the corresponding value
            promotion_value = 0
            if promotion is not None:
                promotion_map = {
                    chess.QUEEN: 1,
                    chess.ROOK: 2,
                    chess.BISHOP: 3,
                    chess.KNIGHT: 4,
                }
                promotion_value = promotion_map[promotion]

            # Calculate the action number
            action = (from_square * 64 * 5) + (to_square * 5) + promotion_value

            return action
        except ValueError:
            return None


class InteractiveEnvironment(ChessEnv):
    def play(self):
        while not self.board.is_game_over():
            self.render()
            action = self.notation_to_action(input("Make a move: "))
            print(self.explain_action(action))
            if action is None:
                continue
            _, reward, done, _ = self.step(action)
            print(f"Reward: {reward}, Done: {done}")

            # for event in pygame.event.get():
            #    if event.type == pygame.QUIT:
            #        pygame.quit()
            #        return
            #    elif event.type == pygame.MOUSEBUTTONDOWN:
            #        if event.button == 1:  # Left mouse button
            #            from_square = self._get_square_from_mouse(event.pos)
            #            to_square = self._get_square_from_mouse(event.pos)
            #            if from_square != to_square:
            #                move = chess.Move(from_square, to_square)
            #                if move in self.board.legal_moves:
            #                    observation, reward, done, _ = self.step(move)
            #                    print(f"Reward: {reward}, Done: {done}")

        self.render()
        result = self.board.result()
        print(f"Game Over. Result: {result}")

    def _get_square_from_mouse(self, pos):
        file, rank = pos[0] // SQUARE_SIZE, 7 - pos[1] // SQUARE_SIZE
        return chess.square(file, rank)


if __name__ == "__main__":
    env = InteractiveEnvironment()
    env.play()
