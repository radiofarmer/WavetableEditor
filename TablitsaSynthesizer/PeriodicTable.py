from WavetableEditor.Wavetable import *


def gaussian(x, m, s, scale=False):
    return (1. / (s * np.sqrt(2 * np.pi)) if scale else 1.) * np.exp(-(x - m) ** 2 / (2 * s ** 2))


names = ["Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
         "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium",
         "Calcium", "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron", "Cobalt", "Nickel", "Copper",
         "Zinc", "Gallium", "Germanium", "Arsenic", "Selenium", "Bromine", "Krypton", "Rubidium", "Strontium",
         "Yttrium", "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium", "Palladium", "Silver",
         "Cadmium", "Indium", "Tin", "Antimony", "Tellurium", "Iodine", "Xenon", "Cesium", "Barium", "Lanthanum",
         "Cerium", "Praseodymium", "Neodymium", "Promethium", "Samarium", "Europium", "Gadolinium", "Terbium",
         "Dysprosium", "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium", "Hafnium", "Tantalum", "Tungsten",
         "Rhenium", "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium", "Lead", "Bismuth", "Polonium",
         "Astatine", "Radon", "Francium", "Radium", "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium",
         "Plutonium", "Americium", "Curium", "Berkelium", "Californium", "Einsteinium", "Fermium", "Mendelevium",
         "Nobelium", "Lawrencium", "Rutherfordium", "Dubnium", "Seaborgium", "Bohrium", "Hassium", "Meitnerium",
         "Darmstadtium", "Roentgenium", "Copernicium", "Nihonium", "Flerovium", "Moscovium", "Livermorium",
         "Tennessine", "Oganesson"]

symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
           'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
           'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
           'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
           'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
           'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
           'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

periods = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
           7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

groups = [1.0, 18.0, 1.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1.0,
          2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1.0, 2.0, 3.0, "La", "La",
          "La", "La", "La", "La", "La", "La", "La", "La", "La", "La", "La", "La", 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
          11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 1.0, 2.0, 3.0, "Ac", "Ac", "Ac", "Ac", "Ac", "Ac", "Ac", "Ac",
          "Ac", "Ac", "Ac", "Ac", "Ac", "Ac", 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
          17.0, 18.0]

oxidation = {'H': [-1, 0, 1],
             'He': [0, 8],
             'Li': [0, 1],
             'Be': [0, 2],
             'B': [0, 3],
             'C': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
             'N': [-3, 0, 3, 5],
             'O': [-2, -1, 0],
             'F': [-1, 0],
             'Ne': [0, 8],
             'Na': [0, 1],
             'Mg': [0, 2],
             'Al': [0, 3],
             'Si': [-4, 0, 4],
             'P': [-3, 0, 3, 5],
             'S': [-2, 0, 2, 4, 6],
             'Cl': [-1, 0, 1, 3, 5, 7],
             'Ar': [0, 8],
             'K': [0, 1],
             'Ca': [0, 2],
             'Sc': [0, 3],
             'Ti': [0, 4],
             'V': [0, 5],
             'Cr': [0, 3, 6],
             'Mn': [0, 2, 4, 7],
             'Fe': [0, 2, 3, 6],
             'Co': [0, 2, 3],
             'Ni': [0, 2],
             'Cu': [0, 1, 2],
             'Zn': [0, 2],
             'Ga': [0, 3],
             'Ge': [-4, 0, 2, 4],
             'As': [-3, 0, 3, 5],
             'Se': [-2, 0, 2, 4, 6],
             'Br': [-1, 0, 1, 3, 5],
             'Kr': [0, 8],
             'Rb': [0, 1],
             'Sr': [0, 2],
             'Y': [0, 3],
             'Zr': [0, 4],
             'Nb': [0, 5],
             'Mo': [0, 4, 6],
             'Tc': [0, 4, 7],
             'Ru': [0, 3, 4],
             'Rh': [0, 3],
             'Pd': [0, 2, 4],
             'Ag': [0, 1],
             'Cd': [0, 2],
             'In': [0, 3],
             'Sn': [-4, 0, 2, 4],
             'Sb': [-3, 0, 3, 5],
             'Te': [-2, 0, 2, 4, 6],
             'I': [-1, 0, 1, 3, 5, 7],
             'Xe': [0, 1, 8],
             'Cs': [0, 1],
             'Ba': [0, 2],
             'La': [0, 3],
             'Ce': [0, 3, 4],
             'Pr': [0, 3],
             'Nd': [0, 3],
             'Pm': [0, 3],
             'Sm': [0, 3],
             'Eu': [0, 3],
             'Gd': [0, 3],
             'Tb': [0, 3],
             'Dy': [0, 3],
             'Ho': [0, 3],
             'Er': [0, 3],
             'Tm': [0, 3],
             'Yb': [0, 3],
             'Lu': [0, 3],
             'Hf': [0, 4],
             'Ta': [0, 5],
             'W': [0, 4, 6],
             'Re': [0, 4],
             'Os': [0, 4],
             'Ir': [0, 3, 4],
             'Pt': [0, 2, 4],
             'Au': [0, 1, 3],
             'Hg': [0, 1, 2],
             'Tl': [0, 1, 3],
             'Pb': [0, 2, 4],
             'Bi': [0, 3],
             'Po': [-2, 0, 2, 4],
             'At': [-1, 0, 1],
             'Rn': [0, 2],
             'Fr': [0, 1],
             'Ra': [0, 2],
             'Ac': [0, 3],
             'Th': [0, 4],
             'Pa': [0, 5],
             'U': [0, 4, 6],
             'Np': [0, 5],
             'Pu': [0, 4],
             'Am': [0, 3],
             'Cm': [0, 3],
             'Bk': [0, 3],
             'Cf': [0, 3],
             'Es': [0, 3],
             'Fm': [0, 3],
             'Md': [0, 3],
             'No': [0, 2],
             'Lr': [0, 3],
             'Rf': [0, 4],
             'Db': [0, 5],
             'Sg': [0, 6],
             'Bh': [0, 7],
             'Hs': [0, 8],
             'Mt': [0, 8],
             'Ds': [0, 8],
             'Rg': [0, 8],
             'Cn': [0, 2],
             'Nh': [0, 8],
             'Fl': [0, 8],
             'Mc': [0, 8],
             'Lv': [0, 8],
             'Ts': [0, 8],
             'Og': [0, 8]}
# idea: maybe add a special "oxidation state" code (a string or something) for "radioactivity"

# Data for additional harmonic series with different periodicities, as determined by oxidation number
# Format is [multiple of fundamental, amplitude scalar, number of cycles required for a whole number]
intervals = {-4: [[5. / 3, 0.8, 3]],
             -3: [[5. / 4, 0.9, 4]],
             -2: [[3. / 4, 0.9, 4]],
             -1: [[1. / 2, 1., 2]],
             0: [[1., 1.00, 1]],
             1: [[3. / 2, 0.9, 2]],
             2: [[1. / 4, 0.6, 4]],
             3: [[2. / 3, 0.75, 3]],
             4: [[4. / 3, 0.85, 3]],
             5: [[1. / 4, 0.75, 4], [3. / 4, 0.75, 4]],
             6: [[5. / 6, 0.7, 6], [1. / 6, 0.7, 6]],
             7: [[1. / 3, 0.8, 6], [2. / 3, 0.8, 6]],
             8: [[1. / 12, 0.5, 12]]}
harmonic_shifts = {
    -4: [4, -1, np.tanh(np.linspace(-1, 6, 100)) / 2 + 0.5],
    -3: [3, -1, np.tanh(np.linspace(-5, 5, 100)) / 2 + 0.5],
    -2: [3, -2, gaussian(np.linspace(0, 1, 100), 0.5, 0.3)],
    -1: [2, -3, np.linspace(0, 1, 100) ** 0.5],
    0: None,
    1: [2, 1, np.linspace(0, 1, 100) ** 0.4],
    2: [3, 5, np.log(np.linspace(1, np.exp(1), 100))],
    3: [2, 5, np.linspace(0, 1, 100) ** 0.5],
    4: [12, 5, np.linspace(0, 1, 100) ** 0.6],
    5: [4, 5, np.linspace(0, 1, 100) ** 0.5],
    6: [6, 7, np.linspace(0, 1, 100) ** 0.5],
    7: [6, 5, np.linspace(0, 1, 100) ** 0.4],
    8: [12, 13, np.linspace(0, 1, 100) ** 0.5]
}

# Amplitude functions, assigned by period.
# x: harmonic number
# p: period of current element
group_funcs = {1: [lambda x, p: x.astype(np.float64) ** (-1 - (p - 1) / 7),
                   None],
               2: [lambda x, p: (1 / x) ** (1 + p / 7) * np.cos(p * np.pi * x / np.max(x)), None],
               3: [lambda x, p: np.exp(((-x / np.max(x)) / (p * 2))), None],
               4: [lambda x, p: ((x + 1) ** 2 / p) % p * np.log((x + 1) / (np.max(x) * p)), None],
               5: [lambda x, p: (np.sqrt(x + p) / (x + 10 * p ** 2) + np.exp(-np.sqrt(x) / p)) ** p + p / 5 * gaussian(
                   x, p ** 2, p), None],
               6: [lambda x, p: np.cos((p * (1 - x / np.max(x))) ** p) / (x / (100 * p)),
                   None],
               7: [lambda x, p: 1 / (x * (np.sin(2 * np.pi * x / p) / 3 + 1)), None],
               8: [lambda x, p: np.exp(-x / np.max(x) * p) * np.cos(8 * p * np.pi * (x / np.max(x)) ** 2),
                   lambda x, p: np.exp(-x[::-1] / np.max(x) * p) * np.cos(8 * p * np.pi * (x[::-1] / np.max(x)) ** 2)],
               9: [lambda x, p: np.exp(-x / np.max(x) * p ** 2) * (
                       0.5 + np.cos(p * np.pi * (p * np.pi * (x / np.max(x))))), None],
               10: [lambda x, p: np.cos(2 * np.pi * x * p) / np.ceil(x / np.max(x) * 10 * p), None],
               11: [lambda x, p: np.cos(np.pi * x / np.max(x) * p ** 2) * (1. / x + (1. / (x + p)) ** p), None],
               12: [lambda x, p: 1 / x + 0.5 * gaussian(x, p * 5, p), None],
               13: [lambda x, p: 1 / np.sqrt(x * 2) * oscillating_signs(x * 2) + np.sum(
                   [1 / (i + 2) * gaussian(x, p * 5 * i, p / i) * (-1 if i % 2 else 1) for i in range(1, p + 1)],
                   axis=0), None],
               14: [lambda x, p: 1 / x * (x % p) * oscillating_signs(x), None],
               15: [lambda x, p: np.cos(p ** 2 * np.pi * np.tanh(x / np.max(x))) * np.exp(-x / (5 * p)), None],
               16: [lambda x, p: np.tanh(1 - np.tanh((x - p * 15) / (p * np.max(x))) ** p / 3) + np.exp(
                   -x / (2 * p)) * oscillating_signs(x),
                    None],
               17: [lambda x, p: np.exp(-x / np.max(x) * p) * (
                       np.cos(2 * np.pi * np.cos((x - 1) / p)) / 2 + 1) * oscillating_signs(x),
                    None],
               18: [lambda x, p: (1 / x) ** (1 + (p - 1) / 7) * (np.floor(x) % 2), None],
               "La": [lambda x, p: 1 / x * np.random.random(x.shape[0]) * p / 7, None],
               "Ac": [lambda x, p: np.exp(-x / np.max(x)) * np.random.random(x.shape[0]) * p / 7, None]}

periodicities = {1: [1, 2],
                 2: [3, 2, ],
                 3: [1, 2, 1.5],
                 4: [0.5, 1, ],
                 5: [0.25, 0.5, 1, ],
                 6: [1 / 3, 0.5, 1],
                 7: [1.5, 1 / 3]}


def lcm(arg1, arg2, *args):
    lcm_ = (arg1 * arg2) // math.gcd(arg1, arg2)
    for argx in args:
        lcm_ = (lcm_ * argx) // math.gcd(lcm_, argx)
    return lcm_


def phase_shift(x, amt, factor=2):
    x %= 1.
    return x ** (1 + (amt if amt >= 0 else 1. / amt) * (factor - 1))


def make_wavetable(h_start, h_end, state, amp_func, period, amp_scale, phase_func, h_shift_func):
    if phase_func is None:
        phase_func = zero_phase
    hm = HarmonicSeries(h_start, h_end, state,
                        amp_func=lambda h, p=period: amp_func(h, p),
                        phase_func=lambda h, p=period: phase_func(h, p) + np.random.random(h.shape[0]) * period/100,
                        h_shift_func=h_shift_func,
                        normalize=1)
    hm *= amp_scale  # scale by the given scale factor, as well as by the period
    return hm


if __name__ == "__main__":
    start = names.index("Hydrogen")

    for sym, name, pd, g in zip(symbols[start:len(oxidation)], names[start:len(oxidation)],
                                periods[start:len(oxidation)], groups[start:len(oxidation)]):
        wt = []
        max_cycles = 1
        # Loop through oxidation state values for the current element
        for state in oxidation[sym]:
            wt.append(Waveform())
            # for s in [intervals[0][0], *intervals[state]]:
            if harmonic_shifts[state] is None:
                shift_base = 1
                shift_fact = 1
                shift_func = None
            else:
                shift_base = harmonic_shifts[state][0]
                shift_fact = harmonic_shifts[state][1]
                shift_curve = harmonic_shifts[state][2]
                shift_func = make_shift_function(shift_curve, shift_base, shift_fact)
            wt[-1].append_series(make_wavetable(1, 200, 1,
                                                group_funcs[g][0],
                                                period=pd,
                                                amp_scale=1.,
                                                phase_func=group_funcs[g][1],
                                                h_shift_func=shift_func))
            wt[-1].append_series(make_wavetable(1, 200, 1 + np.abs(shift_fact) / shift_base,
                                                group_funcs[g][0], period=pd,
                                                amp_scale=2. / shift_base,
                                                phase_func=group_funcs[g][1],
                                                h_shift_func=None))
            max_cycles = abs(lcm(int(max_cycles), int(shift_base)))

        if max_cycles > 12:
            print(name, "had", max_cycles, "cycles")
        IO.export_mipmap_bytes(wt, "Wavetables\\", name + ".wt", max_fps=2 ** 14,
                               cycles_per_level=max_cycles, oversampling=8,
                               levels=[max(2 ** n, 1024) for n in range(14, 3, -1)],
                               limits=[int(2 ** n) for n in range(12, 1, -1)],
                               scale_to=0.5)
        print("Saved wavetable for", name)
