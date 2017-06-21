#!/usr/bin/env python
import pprint
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from collections import namedtuple
from tkinter import *
from tkinter.filedialog import askopenfilename



pp = pprint.PrettyPrinter(indent=4)



color_to_plot_map = {
     'blue': 'b',
      'red': 'r',
    'amber': 'y',
}


class Scale:
    def __init__(self, minor, major):
        self.minor = minor
        self.major = major




def get_options():

    parser = argparse.ArgumentParser(description='View Results')

    parser.add_argument('files', type=str, nargs='+',
                        help='files to be viewed in json format')
    parser.add_argument('--output-only', '-x', action='store_true',
                        help='only generate plots, to not open viewer', default=False)
    parser.add_argument('--well', '-w', action='store_true',
                        help='generate well plots', default=False)
    parser.add_argument('--color', '-c', nargs='?', const='all',
                        help='generate color plots', default=None)
    parser.add_argument('--raw', '-r', action='store_true',
                        help='disable baseline', default=False)
    parser.add_argument('--fixed-scale', '-f', action='store_true',
                        help='Use fixed scale for all graphs PER json file', default=False)
    parser.add_argument('--low-gain', '-l', action='store_true',
                        help='graph using low gain instead of the default high gain', default=False)


    return parser.parse_args()


class SampleData:
    def __init__(self, keyValue):

        # import ipdb; ipdb.set_trace()
        self.code  = keyValue['code']
        self.color = keyValue['color']
        self.cycle = keyValue['cycle']
        self.value = keyValue['value']
        self.well  = keyValue['well']

        if 'gain' in keyValue:
            self.gain = keyValue['gain']
        else:
            self.gain = None

        if 'reference' in keyValue:
            self.reference = keyValue['reference']
        else:
            self.reference = None



def load_data(path):
    all_samples = None

    with open(path, 'r') as content_file:
        content = content_file.read()
        all_samples = json.loads(content, object_hook=SampleData)

    return all_samples



def plot_all_wells(data, save_name=None, fixed_scale=None, file_name=''):
    all_cycles = list(map(lambda x: x.cycle, data))
    f, ax = build_configure_plot('All Wells All Colors', all_cycles)

    plt.title(file_name)

    if fixed_scale is not None:
        plt.ylim([fixed_scale.minor, fixed_scale.major])

    # Extract only color, reduce to distinct set
    colors = set(map(lambda x: x.color, data))
    wells = set(map(lambda x: x.well, data))

    plotList = []
    for w in wells:
        for c in colors:
            # get data set for specific well and color
            samples = list(filter(lambda x: x.color == c and x.well == w, data))

            samples = sorted(samples, key=lambda x: x.cycle)

            value = list(map(lambda x: x.value, samples))
            cycle = list(map(lambda x: x.cycle, samples))
            label = 'Well {0} {1}'.format(w,c)

            p, = plt.plot(cycle, value, label=label)

            plotList.append(p)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(handles=plotList, loc='center left', bbox_to_anchor=(1, 0.5))

    if save_name is not None:
        f.saveFigure(save_name)



def gains_are_present(data):
    matches = list(filter(lambda x: x.gain == 'high' or x.gain == 'low', data))
    return len(matches) > 0



def plot_well_all_colors(well, data, save_name=None, fixed_scale=None):
    all_cycles = list(map(lambda x: x.cycle, data))

    title = 'Well {0} All Colors'.format(well)
    f, ax = build_configure_plot(title, all_cycles)

    if fixed_scale is not None:
        plt.ylim([fixed_scale.minor, fixed_scale.major])

    # Extract only color, reduce to distinct set
    colors = set(map(lambda x: x.color, data))

    plotList = []
    for c in colors:
        # get data set for specific well and color
        samples = list(filter(lambda x: x.color == c and x.well == well, data))

        samples = sorted(samples, key=lambda x: x.cycle)

        value = list(map(lambda x: x.value, samples))
        cycle = list(map(lambda x: x.cycle, samples))

        plot_color = color_to_plot_map[c]

        p, = plt.plot(cycle, value, plot_color, label=c)

        plotList.append(p)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(handles=plotList, loc='center left', bbox_to_anchor=(1, 0.5))


    if save_name is not None:
        f.saveFigure(save_name)



def plot_color_all_wells(color, data, save_name=None, fixed_scale=None):
    all_cycles = list(map(lambda x: x.cycle, data))

    title = 'Color {0} All wells'.format(color.upper())
    f, ax = build_configure_plot(title, all_cycles)

    wells = set(map(lambda x: x.well, data))

    if fixed_scale is not None:
        plt.ylim([fixed_scale.minor, fixed_scale.major])

    plotList = []
    for w in wells:
        # get data set for specific well and color
        samples = list(filter(lambda x: x.color == color and x.well == w, data))

        samples = sorted(samples, key=lambda x: x.cycle)

        value = list(map(lambda x: x.value, samples))
        cycle = list(map(lambda x: x.cycle, samples))

        p, = plt.plot(cycle, value, label=str(w))

        plotList.append(p)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(handles=plotList, loc='center left', bbox_to_anchor=(1, 0.5))

    if save_name is not None:
        f.saveFigure(save_name)



def plot_per_color(data, save_name=None, fixed_scale=None):
    for c in set(map(lambda x: x.color, data)):
        plot_color_all_wells(c, data, save_name, fixed_scale=fixed_scale)


def plot_per_well(data, save_name=None, fixed_scale=None):
    for w in set(map(lambda x: x.well, data)):
        plot_well_all_colors(w, data, save_name, fixed_scale=fixed_scale)



def build_configure_plot(title, cycles=[]):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)

    f.suptitle(title, fontsize=14, fontweight='bold')
    plt.xlabel('Cycle')
    plt.ylabel('Intensity')
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    if len(cycles) > 1:
        plt.xticks(np.arange(min(cycles), max(cycles)+1, 5.0))
        ax.set_xticks(np.arange(min(cycles), max(cycles)+1, 1.0), minor=True)

    return f, ax


def baseline_all_wells(data):
    colors = set(map(lambda x: x.color, data))
    wells = set(map(lambda x: x.well, data))

    for w in wells:
        for c in colors:
            # get data set for specific well and color
            samples = list(filter(lambda x: x.color == c and x.well == w, data))

            samples = sorted(samples, key=lambda x: x.cycle)

            value = np.array(list(map(lambda x: x.value, samples)))
            cycle = np.array(list(map(lambda x: x.cycle, samples)))

            max_cycle = max(list(map(lambda x: x.cycle, samples)))


            if (max_cycle < 30):
                continue

            # import ipdb; ipdb.set_trace()

            baseline = np.polyfit(cycle[5:20], value[5:20], 1)

            apply_baseline(samples, baseline)



def apply_baseline(sample, baseline):
    gain = baseline[0]
    offset = baseline[1]

    for s in sample:
        s.value -= s.cycle * gain + offset


def get_global_scale(files, raw, gain):
    low = 0.0
    high = 0.0
    for f in files:
        data = load_data(f)

        data = list(filter(lambda x: x.gain == gain, data))

        if raw is False:
            baseline_all_wells(data)

        values = list(map(lambda x: x.value, data))

        min_file = min(values)
        max_file = max(values)

        low = min(min_file, low)
        high = max(max_file, high)

    low = low - (low * 0.1)
    high = high + (high * 0.1)

    return Scale(low, high)


if __name__ == "__main__":
    options = get_options()

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    file = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(file)

    gain = 'high'
    if options.low_gain:
        gain = 'low'

    scale = None
    if options.fixed_scale:
        scale = get_global_scale(options.files, options.raw, gain)

    #for file in options.files:
     #   data = load_data(file)


        if gains_are_present(data):
            data = list(filter(lambda x: x.gain == gain, data))
            file = "File:{0} Gain:{1}".format(file, gain)

        if options.raw is False:
            baseline_all_wells(data)


        plot_all_wells(data, file_name=file, fixed_scale=scale)

        if options.well:
            plot_per_well(data, fixed_scale=scale)

        if 'all' is options.color:
            plot_per_color(data, fixed_scale=scale)
        elif options.color in color_to_plot_map:
            plot_color_all_wells(options.color, data, fixed_scale=scale)

    if not options.output_only:
        plt.show()
