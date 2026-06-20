class Goal:
    def can_use(self):
        return False

    def can_continue_to_use(self):
        return self.can_use()

    def start(self):
        pass

    def stop(self):
        pass

    def tick(self):
        pass

class GoalSelector:
    def __init__(self):
        self.goals = []
        self.active_goals = []

    def add_goal(self, priority, goal):
        self.goals.append({'priority': priority, 'goal': goal})
        self.goals.sort(key=lambda x: x['priority'])

    def tick(self):
        # Stop goals that can't continue
        for entry in self.active_goals[:]:
            if not entry['goal'].can_continue_to_use():
                entry['goal'].stop()
                self.active_goals.remove(entry)

        # Start new goals
        for entry in self.goals:
            if entry not in self.active_goals and entry['goal'].can_use():
                # Check priority conflicts (simplified: just run all non-conflicting or let them run)
                # In full implementation, we'd check mutex bits.
                entry['goal'].start()
                self.active_goals.append(entry)

        # Tick active goals
        for entry in self.active_goals:
            entry['goal'].tick()
