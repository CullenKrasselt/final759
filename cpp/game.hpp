/**
 * game.hpp — Minesweeper game engine (header-only).
 *
 * Board is a flat row-major array of size rows*cols.
 * Cell values:
 *   CELL_UNKNOWN (-1) : not yet revealed
 *   CELL_MINE   (-2) : revealed mine (game over)
 *   0-8              : revealed cell, value = adjacent mine count
 *
 * Mines are placed on the first reveal, guaranteeing the opening
 * cell and all its neighbours are safe.
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

static constexpr int CELL_UNKNOWN = -1;
static constexpr int CELL_MINE   = -2;

class MinesweeperGame {
public:
    const int rows, cols, total_mines;
    std::vector<int> board;   // flat row-major: board[r*cols + c]
    bool game_over = false;
    bool won       = false;

    MinesweeperGame(int rows, int cols, int total_mines,
                    unsigned seed = std::random_device{}())
        : rows(rows), cols(cols), total_mines(total_mines),
          board(rows * cols, CELL_UNKNOWN),
          _mines(rows * cols, false),
          _rng(seed),
          _initialized(false)
    {
        if (total_mines >= rows * cols)
            throw std::invalid_argument("Too many mines for the board size.");
    }

    /**
     * Reveal cell (r, c).
     * Returns false if a mine was hit (game over), true otherwise.
     */
    bool reveal(int r, int c) {
        if (game_over || board[_idx(r, c)] != CELL_UNKNOWN)
            return true;
        if (!_initialized)
            _place_mines(r, c);
        if (_mines[_idx(r, c)]) {
            board[_idx(r, c)] = CELL_MINE;
            game_over = true;
            return false;
        }
        _flood_reveal(r, c);
        _check_win();
        return true;
    }

    void print_board(bool reveal_all = false) const {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int v = board[_idx(r, c)];
                if (v == CELL_UNKNOWN)
                    std::cout << (reveal_all && _mines[_idx(r, c)] ? 'X' : '.');
                else if (v == CELL_MINE) std::cout << '*';
                else if (v == 0)         std::cout << ' ';
                else                     std::cout << v;
            }
            std::cout << '\n';
        }
    }

private:
    std::vector<bool> _mines;
    std::mt19937      _rng;
    bool              _initialized;

    int _idx(int r, int c) const { return r * cols + c; }

    template<typename F>
    void _for_neighbors(int r, int c, F&& fn) const {
        for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols)
                    fn(nr, nc);
            }
    }

    void _place_mines(int safe_r, int safe_c) {
        std::unordered_set<int> safe;
        safe.insert(_idx(safe_r, safe_c));
        _for_neighbors(safe_r, safe_c, [&](int nr, int nc) {
            safe.insert(_idx(nr, nc));
        });

        std::vector<int> candidates;
        candidates.reserve(rows * cols - (int)safe.size());
        for (int i = 0; i < rows * cols; ++i)
            if (!safe.count(i)) candidates.push_back(i);

        std::shuffle(candidates.begin(), candidates.end(), _rng);
        for (int i = 0; i < total_mines; ++i)
            _mines[candidates[i]] = true;
        _initialized = true;
    }

    int _adj_mine_count(int r, int c) const {
        int count = 0;
        _for_neighbors(r, c, [&](int nr, int nc) {
            count += _mines[_idx(nr, nc)] ? 1 : 0;
        });
        return count;
    }

    void _flood_reveal(int start_r, int start_c) {
        std::vector<int> stack = {_idx(start_r, start_c)};
        std::vector<bool> seen(rows * cols, false);
        while (!stack.empty()) {
            int i = stack.back(); stack.pop_back();
            if (seen[i]) continue;
            seen[i] = true;
            int r = i / cols, c = i % cols;
            int count = _adj_mine_count(r, c);
            board[i] = count;
            if (count == 0) {
                _for_neighbors(r, c, [&](int nr, int nc) {
                    int ni = _idx(nr, nc);
                    if (board[ni] == CELL_UNKNOWN && !seen[ni])
                        stack.push_back(ni);
                });
            }
        }
    }

    void _check_win() {
        int unknown = 0, confirmed_mines = 0;
        for (int v : board) {
            if (v == CELL_UNKNOWN)  ++unknown;
            if (v == CELL_MINE)     ++confirmed_mines;
        }
        if (unknown == total_mines - confirmed_mines) {
            won = true;
            game_over = true;
        }
    }
};
