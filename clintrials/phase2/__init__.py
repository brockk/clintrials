__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


__all__ = ["simple"]


class Phase2EffToxBase:

    def __init__(self):
        pass

    def size(self):
        return len(self.cases)

    def pretreated_statuses(self):
        return [case[0] for case in self.cases]

    def mutation_statuses(self):
        return [case[1] for case in self.cases]

    def efficacies(self):
        return [case[2] for case in self.cases]

    def toxicities(self):
        return [case[3] for case in self.cases]