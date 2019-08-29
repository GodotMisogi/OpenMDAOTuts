import openmdao.api as om

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
        
        self.declare_partials('I', 'V_in', method='fd')
        self.declare_partials('I', 'V_out', method='fd')
        
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
        
        self.declare_partials('I', 'V_in', method='fd')
        self.declare_partials('I', 'V_out', method='fd')
        
    def compute(self, inputs, outputs):
        deltaV = inputs['V_in'] - inputs['V_out']
        Is = self.options['Is']
        Vt = self.options['Vt']
        outputs['I'] = Is* (np.exp(deltaV / Vt) - 1)

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
            residuals['V'] += inputs[f'I_out:{i_conn}']

class Circuit(om.Group):
    
    def setup(self):
        self.add_subsystem('n1', Node(n_in=1, n_out=2), promotes_inputs=[('I_in:0', 'I_in')])
        self.add_subsystem('n2', Node()) # Leaving defaults
        
        self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
        self.add_subsystem('R2', Resistor(R=10000.))
        self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])
        
        # USE PROMOTES
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
        
prob = om.Problem()
model = prob.model

model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
model.add_subsystem('circuit', Circuit())

model.connect('source.I', 'circuit.I_in')
model.connect('ground.V', 'circuit.Vg')

prob.setup()

# Initial values
prob['circuit.n1.V'] = 10.
prob['circuit.n2.V'] = 1.

prob.run_model()