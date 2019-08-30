from __future__ import division, print_function
import openmdao.api as om
import numpy as np

class Resistor(om.ExplicitComponent):
    """
    Computes current across a resistor using Ohm's law.
    """
    
    def initialize(self):
        self.options.declare('R', default=1., desc='Resistance in Ohms')
        
    def setup(self):
        self.add_input('V_in', units='V')
        self.add_input('V_out', units='V')
        self.add_output('I', units='A')
        
        # Finite differences
#         self.declare_partials('I', 'V_in', method='fd')
#         self.declare_partials('I', 'V_out', method='fd')
        
        # Analytic derivatives - Partial derivatives are constant, so their values can be assigned in setup
        R = self.options['R']
        self.declare_partials('I', 'V_in', val=1/R)
        self.declare_partials('I', 'V_out', val=-1/R)
        
    def compute(self, inputs, outputs):
        deltaV = inputs['V_in'] - inputs['V_out']
        outputs['I'] = deltaV / self.options['R']
        
class Diode(om.ExplicitComponent):
    """
    Computes current across a diode using the Shockley diode equation.
    """
    
    def initialize(self):
        self.options.declare('Is', default=1e-15, desc='Saturation current in Amps')
        self.options.declare('Vt', default=0.025875, desc='Thermal voltage in Volts')
        
    def setup(self):
        self.add_input('V_in', units='V')
        self.add_input('V_out', units='V')
        self.add_output('I', units='A')
        
        # Finite differences
#         self.declare_partials('I', 'V_in', method='fd')
#         self.declare_partials('I', 'V_out', method='fd')
        
        # Analytic derivatives declaration
        self.declare_partials('I', 'V_in')
        self.declare_partials('I', 'V_out')
        
    def compute(self, inputs, outputs):
        deltaV = inputs['V_in'] - inputs['V_out']
        Is = self.options['Is']
        Vt = self.options['Vt']
        outputs['I'] = Is * (np.exp(deltaV / Vt) - 1)
    
    def compute_partials(self, inputs, J):
        deltaV = inputs['V_in'] - inputs['V_out']
        Is = self.options['Is']
        Vt = self.options['Vt']
        I = Is * np.exp(deltaV / Vt)
        
        J['I', 'V_in'] = I/Vt
        J['I', 'V_out'] = -I/Vt
        
class Node(om.ImplicitComponent):
    """
    Computes voltage residual across a node based on incoming and outgoing current.
    """
    
    def initialize(self):
        self.options.declare('n_in', default=1, types=int, desc='Number of connections with + assumed in')
        self.options.declare('n_out', default=1, types=int, desc='Number of current connections + assumed out')
        
    def setup(self):
        self.add_output('V', val=5., units='V')
        
        for i in range(self.options['n_in']):
            i_name = f'I_in:{i}'
            self.add_input(i_name, units='A')
            
        for i in range(self.options['n_out']):
            i_name = f'I_out:{i}'
            self.add_input(i_name, units='A')
            
        # Note: We don't declare any partials wrt `V` here, because the residual doesn't directly depend on it
        self.declare_partials('V', 'I*', method='fd')
        
    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['V'] = 0.
        for i_conn in range(self.options['n_in']):
            residuals['V'] += inputs[f'I_in:{i_conn}']
        for i_conn in range(self.options['n_out']):
            residuals['V'] -= inputs[f'I_out:{i_conn}']

class Circuit(om.Group):
    
    def setup(self):
        self.add_subsystem('n1', Node(n_in=1, n_out=2), promotes_inputs=[('I_in:0', 'I_in')])
        self.add_subsystem('n2', Node()) # Leaving defaults
        
        self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
        self.add_subsystem('R2', Resistor(R=10000.))
        self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])
        
        # Not possible to use promotes because of weird variable naming?
        self.connect('n1.V', ['R1.V_in', 'R2.V_in'])
        self.connect('R1.I', 'n1.I_out:0')
        self.connect('R2.I', 'n1.I_out:1')
        
        self.connect('n2.V', ['R2.V_out', 'D1.V_in'])
        self.connect('R2.I', 'n2.I_in:0')
        self.connect('D1.I', 'n2.I_out:0')
        
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = om.DirectSolver()

p = om.Problem()
model = p.model

model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
# Replacing the fixed current source with a BalanceComp to represent a fixed voltage source
model.add_subsystem('batt', om.IndepVarComp('V', 1.5, units='V'))
bal = model.add_subsystem('batt_balance', om.BalanceComp())
bal.add_balance('I', units='A', eq_units='V')

model.add_subsystem('circuit', Circuit())
model.add_subsystem('batt_deltaV', om.ExecComp('dV = V1 - V2', V1={'units':'V'}, V2={'units':'V'}, dV={'units':'V'}))

# The current into the circuit is now the output state from the batt_balance comp
model.connect('batt_balance.I', 'circuit.I_in')
model.connect('ground.V', ['circuit.Vg', 'batt_deltaV.V2'])
model.connect('circuit.n1.V', 'batt_deltaV.V1')

# Set the LHS and RHS for the battery residual
model.connect('batt.V', 'batt_balance.rhs:I')
model.connect('batt_deltaV.dV', 'batt_balance.lhs:I')

p.setup()

# Change the circuit solver to RunOnce because we're going to converge at the top level of the model with Newton instead
p.model.circuit.nonlinear_solver = om.NonlinearRunOnce()
p.model.circuit.linear_solver = om.LinearRunOnce()

# Put Newton at the top so it can also converge the new BalanceComp residual
newton = p.model.nonlinear_solver = om.NewtonSolver()
p.model.linear_solver = om.DirectSolver()
newton.options['iprint'] = 2
newton.options['maxiter'] = 20
newton.options['solve_subsystems'] = True
newton.linesearch = om.ArmijoGoldsteinLS()
newton.linesearch.options['maxiter'] = 10
newton.linesearch.options['iprint'] = 2

# Initial guesses
p['circuit.n1.V'] = 9.8
p['circuit.n2.V'] = .7

p.run_model()
print(p['circuit.n1.V'], p['circuit.n2.V'], p['circuit.R1.I'], p['circuit.R2.I'], p['circuit.D1.I'])