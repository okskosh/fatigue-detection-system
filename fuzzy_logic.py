import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FatigueSys:
	def __init__(self):
		# new Antecedent/Consequent objects hold universe variables and membership functions
		eye_state = ctrl.Antecedent(np.arange(0, 1, 0.001), 'Eye state')
		mouth_state = ctrl.Antecedent(np.arange(0, 2, 0.001), 'Mouth state')
		self.driver_state = ctrl.Consequent(np.arange(0, 100, 1), 'Driver state')

		eye_state['closed'] = fuzz.trapmf(eye_state.universe, [0, 0, 0.20, 0.35])
		eye_state['half-open'] = fuzz.trapmf(eye_state.universe, [0.20, 0.40, 0.60, 0.80])
		eye_state['open'] = fuzz.trapmf(eye_state.universe, [0.55, 0.80, 1.00, 1.00])

		mouth_state['closed'] = fuzz.trapmf(mouth_state.universe, [0, 0, 0.15, 0.40])
		mouth_state['half-open'] = fuzz.trapmf(mouth_state.universe, [0.15, 0.35, 0.60, 0.90])
		mouth_state['open'] = fuzz.trapmf(mouth_state.universe, [0.70, 0.90, 2.00, 2.00])

		self.driver_state['fatigued'] = fuzz.trapmf(self.driver_state.universe, [0, 0, 20, 50])
		self.driver_state['sluggish'] = fuzz.trapmf(self.driver_state.universe, [20, 50, 70, 90])
		self.driver_state['wakeful'] = fuzz.trapmf(self.driver_state.universe, [70, 90, 100, 100])

		rule1 = ctrl.Rule(eye_state['open'] | mouth_state['closed'], self.driver_state['wakeful'])
		rule2 = ctrl.Rule(eye_state['closed'] | mouth_state['open'], self.driver_state['fatigued'])
		rule3 = ctrl.Rule(eye_state['half-open'] | mouth_state['half-open'], self.driver_state['fatigued'])
		rule4 = ctrl.Rule(eye_state['open'] | mouth_state['half-open'], self.driver_state['wakeful'])
		rule5 = ctrl.Rule(eye_state['half-open'] | mouth_state['closed'], self.driver_state['sluggish'])
		rule6 = ctrl.Rule(eye_state['open'] | mouth_state['open'], self.driver_state['sluggish'])

		fatigue_sys_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule3, rule4, rule5, rule6])
		self.fatigue_sys = ctrl.ControlSystemSimulation(fatigue_sys_ctrl)

	def compute_inference(self, ear, mar):
		self.fatigue_sys.input['Eye state'] = ear
		self.fatigue_sys.input['Mouth state'] = mar

		self.fatigue_sys.compute()

		return self.fatigue_sys.output['Driver state']

	def interpret_membership(self, value):
		terms = {}
		prob = fuzz.interp_membership(np.arange(0, 100, 1), self.driver_state['fatigued'].mf, value)
		terms['fatigued'] = prob
		prob = fuzz.interp_membership(np.arange(0, 100, 1), self.driver_state['sluggish'].mf, value)
		terms['sluggish'] = prob
		prob = fuzz.interp_membership(np.arange(0, 100, 1), self.driver_state['wakeful'].mf, value)
		terms['wakeful'] = prob

		return max(terms, key=terms.get)
