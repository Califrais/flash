from collections import defaultdict


class History:
    """A class to manage the history along iterations of a solver

    Parameters
    ----------
    minimum_col_width : `int`, default=8
        Minimum of the column width for printing history

    print_order : `list`, default=None
        Gives in order the elements to be printed in print_history.
        If None, then print_order = ["n_iter", "obj", "rel_obj"] which is the
        common information we want to print for a solver : current iteration
        number, current objective function value, and current relative objective
        to monitor convergence
    """

    def __init__(self, minimum_col_width=8, print_order=None):
        if print_order is None:
            print_order = ["n_iter", "obj", "rel_obj"]
        self.minimum_col_width = minimum_col_width
        self.print_order = print_order

        # Instantiate values of the history
        self.clear()

        # Default print style of history values. Default is %.2e
        print_style = defaultdict(lambda: "%.2e")
        print_style["n_iter"] = "%d"
        print_style["obj"] = "%g"
        self.print_style = print_style

        # Attributes that will be instantiated afterwards
        self.values = None
        self.n_iter = None
        self.col_widths = None

    def clear(self):
        """Reset history values"""
        self.values = defaultdict(list)

    def update(self, **kwargs):
        """Update the history along the iterations
        """
        n_iter = kwargs["n_iter"]
        self.n_iter = n_iter
        history = self.values
        for key, val in kwargs.items():
            if key == 'theta':
                for key_, val_ in val.items():
                    history[key_].append(val_)
            else:
                history[key].append(val)

    def print_history(self):
        """Verbose for the current line of history regarding print_order
        """
        values = self.values
        n_iter = self.n_iter
        print_order = self.print_order
        # If this is the first iteration, plot the history's column names
        # regarding print_order
        if n_iter == 0:
            min_width = self.minimum_col_width
            line = ' | '.join([name.center(min_width) for name in
                               print_order if name in values])
            names = [name.center(min_width) for name in print_order]
            self.col_widths = list(map(len, names))
            print(line)

        col_widths = self.col_widths
        print_style = self.print_style
        line = ' | '.join([(print_style[name] % values[name][-1])
                          .rjust(col_widths[i])
                           for i, name in enumerate(print_order)
                           if name in values])
        print(line)
