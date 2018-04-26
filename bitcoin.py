import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.widgets import TextBox
from scipy.interpolate import interp1d
import csv
import numpy as np
from scipy import stats
import warnings
from matplotlib.lines import Line2D
warnings.simplefilter('ignore', np.RankWarning)
import threading
from time import sleep
import time
import os
import gdax
import requests
from matplotlib import dates
import datetime

class analyze:
    filename = 0
    input_graph = 0
    output_graph = 0
    fig = 0
    plt = 0

    points = 0
    index = 0
    count = 0
    time_frame = 0

    timestamps = []
    xs = []
    ys = []

    input_xs = []
    input_ys = []
    input_times = []

    interpolated_xs = []
    interpolated_ys = []

    linear_slope = 0
    linear_intercept = 0
    r_value = 0
    p_value = 0
    std_err = 0
    linear_xs = []
    linear_ys = []

    order = 0
    poly_xs = []
    poly_ys = []
    poly_zs = 0
    poly_f = 0

    press_loc = 0
    press_index = 0
    release_loc = 0
    release_index = 0
    update_output_graph_flag = 0
    update_input_graph_flag = 0
    real_time_flag = 0
    hold_program_counter = 0
    close_program_flag = 0
    background_task = 0
    input_graph_overlay_position = 0
    timeout_time_interval = 0
    sample_time_interval = 0

    # Textbox
    filename_textbox = 0
    timeframe_textbox = 0
    sample_interval_textbox = 0

    def __init__(self):
        self.points = 0
        self.index = 0

        self.timestamps = []
        self.xs = []
        self.ys = []
        self.input_xs = []
        self.input_ys = []
        self.input_times = []

        self.count = 0
        self.timeout_time_interval = 10
        self.sample_time_interval = 5
        self.time_frame_min = 1
        self.__set_count__()

        self.interpolated_xs = []
        self.interpolated_ys = []

        self.linear_slope = 0
        self.linear_intercept = 0
        self.r_value = 0
        self.p_value = 0
        self.std_err = 0
        self.linear_xs = self.input_xs
        self.linear_ys = []

        self.order = 5
        self.poly_xs = self.input_xs
        self.poly_ys = []
        self.poly_zs = 0
        self.poly_f = 0

        self.press_loc = 0
        self.press_index = 0
        self.release_loc = 0
        self.release_index = 0
        self.update_output_graph_flag = 0
        self.update_input_graph_flag = 0
        self.hold_program_counter = 0
        self.close_program_flag = 0
        self.real_time_flag = 1
        self.input_graph_overlay_position = 0


        self.filename = "input_filename"
        self.fig = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        self.input_graph = self.fig.add_subplot(2, 1, 1)
        self.output_graph = self.fig.add_subplot(2, 1, 2)

        self.input_graph.grid()
        self.output_graph.grid()


        self.input_graph.clear()
        self.output_graph.clear()
        #self.input_graph.set_ylim(min(self.ys) - 3, max(self.ys) + 3)
        #self.input_graph.set_xlim(self.xs[0], self.xs[-1])
        #self.output_graph.set_ylim(min(self.ys) - 3, max(self.ys) + 3)
        #self.output_graph.set_xlim(self.xs[0], self.xs[-1])

        #hfmt = dates.DateFormatter('%m/%d %H:%M:%S')
        #self.input_graph.xaxis.set_major_formatter(hfmt)
        #self.output_graph.xaxis.set_major_formatter(hfmt)

        self.input_graph.plot(self.xs, self.ys, '-', c='gray', label='original', linewidth=1.0)
        self.input_graph.plot(self.xs, self.ys, 'o', c='black', label='input', linewidth=1.0)
        self.input_graph.set_title("Input Data")
        self.input_graph.set_ylabel("Price (USD)")


        # input_graph.xaxis.set_major_locator(dates.SecondLocator())

        self.input_graph.legend()

        #self.__run_interpolation__()
        #self.__generate_linear_equation__()
        #self.__run_linear_equation__()
        #self.__generate_polynomial_equation__(5)
        #self.__run_polynomial_equation__()

        self.output_graph.plot(self.interpolated_xs, self.interpolated_ys, c="black", label="Interpolated", linewidth=1.0)
        self.output_graph.plot(self.linear_xs, self.linear_ys, c="cyan", label="Linear Regress", linewidth=1.0)
        self.output_graph.plot(self.poly_xs, self.poly_ys, c='blue', label='Poly Fit', linewidth=1.0)
        self.output_graph.set_title("Output Data")
        self.output_graph.set_xlabel("Time (Sec)")
        self.output_graph.set_ylabel("Price (USD)")
        self.output_graph.legend()

        cid = self.fig.canvas.mpl_connect('button_press_event', self.button_press)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.button_release)
        #cid = self.fig.canvas.mpl_connect('motion_notify_event', self.input_graph_overlay_position_func)

        #                       x     y     w     h
        self.filename_textbox = "{}".format(self.filename)
        axfilebox = plt.axes([0.10, 0.20, 0.15, 0.03])
        file_text_box = TextBox(axfilebox, 'Filename:', initial=self.filename_textbox)
        file_text_box.on_submit(self.submit)

        axSave = plt.axes([0.10, 0.15, 0.15, 0.03])
        bSave = Button(axSave, 'Save')
        bSave.on_clicked(self.on_save_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_save)

        axLoad = plt.axes([0.10, 0.10, 0.15, 0.03])
        bLoad = Button(axLoad, 'Load')
        bLoad.on_clicked(self.on_load_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_load)


        axAverage = plt.axes([0.26, 0.20, 0.15, 0.03])
        bAverage = Button(axAverage, 'Average')
        bAverage.on_clicked(self.on_average_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_average)

        axSmooth = plt.axes([0.26, 0.15, 0.15, 0.03])
        bSmooth = Button(axSmooth, 'Smooth')
        bSmooth.on_clicked(self.on_smooth_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_smooth)

        axDelete = plt.axes([0.26, 0.10, 0.15, 0.03])
        bDelete = Button(axDelete, 'Delete')
        bDelete.on_clicked(self.on_delete_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_delete)


        axRT = plt.axes([0.42, 0.20, 0.15, 0.03])
        bRT = Button(axRT, 'Live')
        bRT.on_clicked(self.on_rt_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_rt)

        axClear = plt.axes([0.42, 0.15, 0.15, 0.03])
        bClear = Button(axClear, 'Clear')
        bClear.on_clicked(self.on_clear_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_clear)


        axHold = plt.axes([0.58, 0.20, 0.15, 0.03])
        bHold = Button(axHold, 'Hold')
        bHold.on_clicked(self.on_hold_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_hold)

        axClose = plt.axes([0.58, 0.15, 0.15, 0.03])
        bClose = Button(axClose, 'Close')
        bClose.on_clicked(self.on_close_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_close)

        self.timeframe_textbox = "{}".format(self.time_frame_min)
        axtimeframebox = plt.axes([0.90, 0.20, 0.05, 0.03])
        time_frame_text_box = TextBox(axtimeframebox, 'Span (min):', initial=self.timeframe_textbox)
        time_frame_text_box.on_submit(self.time_frame_submit)

        self.sample_interval_textbox = "{}".format(self.sample_time_interval)
        axsampleintervalbox = plt.axes([0.90, 0.15, 0.05, 0.03])
        sample_interval_text_box = TextBox(axsampleintervalbox, 'Sample Interval (sec):', initial=self.sample_interval_textbox)
        sample_interval_text_box.on_submit(self.sample_interval_submit)

        self.background_task = threading.Thread(target=self.program_loop, args=())
        self.background_task = threading.Thread(target=self.program_loop, args=())
        self.background_task.daemon = True
        self.background_task.start()
        ani = animation.FuncAnimation(self.fig, self.check_graph, interval=1000)

        plt.xticks(rotation='vertical')
        plt.subplots_adjust(bottom=.3)
        plt.show()

    #def input_graph_overlay_position_func(self, event):
    #    if event.y < 340 or event.y > 390 or event.x > 863 or event.x < 123:
    #        self.hold_program_counter = 0
    #        return 0
    #    self.input_graph_overlay_position = 0
    #    if len(self.times) <= 1:
    #        self.hold_program_counter = 0
    #        return 0
    #    raw_x_data = event.xdata
    #    raw_y_data = event.ydata
    #    index = np.searchsorted(self.times, [raw_x_data])[0]
    #    input_x_data = self.times[index]
    #    input_y_data = self.ys[index]
    #    self.input_graph_overlay_position = (input_x_data, input_y_data)

    def program_loop(self):
        t_start = time.time()
        while (not self.close_program_flag):
            if not self.hold_program_counter:
                self.new_data()
                if self.real_time_flag:
                    self.real_time_data_func()
            sleep(self.sample_time_interval)
            t_end = time.time()
            t_diff = round(t_end - t_start, 2)
            print("{}. {}: Count: {} TimeFrame: {}".format(self.points, t_diff, self.count, self.time_frame_min*60))

    def new_data(self):
        self.update_input_graph_flag = 0
        if self.points >= self.count:
            self.points = self.points - 1
            self.index = self.index - 1
            self.timestamps = np.delete(self.timestamps, 0)
            self.xs = np.delete(self.xs, 0)
            self.ys = np.delete(self.ys, 0)

        now = datetime.datetime.now()
        date = dates.date2num(now)
        x = date
        y = self.__get_order_book__()
        print("\nCurrent Price === {} === t={} ===".format(y, x))
        if y == -1:
            print("Price readout error! Trying again.")
            while y == -1:
                sleep(self.timeout_time_interval)
                y = self.__get_order_book__()
                print("\nCurrent Price === {} === t={} ===".format(y, x))

        self.ys = np.append(self.ys, y)
        self.timestamps = np.append(self.timestamps, x)
        self.xs = np.append(self.xs, len(self.timestamps))
        self.points = self.points + 1
        self.index = self.index + 1
        self.update_input_graph_flag = 1

    def __set_count__(self):
        #self.sample_time_interval = 2
        if self.time_frame_min < 1:
            print("Time frame is less than 1 min")
            exit(1)
        ys = [2, 10, 20, 30, 48, 97, 146, 194, 244, 245, 294, 343, 393, 442, 492, 541, 591, 639, 687, 737, 786, 836,
              885, 934, 982]
        xs = [3.78, 13.19, 25.46, 38.13, 60.20, 119.93, 180.25, 239.98, 300.09, 301.26, 360.64, 420.77, 480.90, 540.11,
              600.13, 659.83, 720.65, 780.78, 840.50, 900.84, 960.11, 1020.25, 1080.58, 1140.65, 1200.08]
        input_xs = np.asarray(xs)
        input_xs = np.multiply(self.sample_time_interval, input_xs)
        if (self.time_frame_min * 60) < input_xs[0]:
            print("Time frame is less than listed or count two")
            exit(1)
        input_ys = np.asarray(ys)
        poly_zs = np.polyfit(input_xs, input_ys, 7)
        poly_f = np.poly1d(poly_zs)
        self.count = int(poly_f(self.time_frame_min * 60))
        if self.count < 2:
            print("Count need to be alteast 2")
            exit(1)

    def __get_order_book__(self):
        try:
            product_id = "BTC-USD"
            public_client = gdax.PublicClient()
            order_book = public_client.get_product_order_book(product_id)
            bids = order_book['bids']
            market_price_bid = float(bids[0][0])
            val_y = round(market_price_bid, 3)
            return val_y
        except KeyError:
            print("Error: Key Error")
            return -1
        except requests.ReadTimeout:
            print("Error: ReadTimeout Error")
            return -1
        except requests.ConnectionError:
            print("Error: Connection Error")
            return -1
        except requests.ConnectTimeout:
            print("Error: Connection Timeout")
            return -1

    def sample_interval_submit(self, text):
        initial = self.sample_time_interval
        try:
            self.sample_time_interval = int("{}".format(text))
        except ValueError:
            print("Value Error")
            self.sample_time_interval = initial
            return 0
        self.__set_count__()

    def time_frame_submit(self, text):
        initial = self.time_frame_min
        try:
            self.time_frame_min = int("{}".format(text))
        except ValueError:
            print("Value Error")
            self.time_frame_min = initial
            return 0
        self.__set_count__()

    def submit(self, text):
        self.filename_textbox = "{}.csv".format(text)
        print(self.filename_textbox)

    def on_close(self, event):
        pass
    def on_close_clicked(self, event):
        self.close_program_flag = 1

    def on_average(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            return 0
        self.hold_program_counter = 1
    def on_average_clicked(self, event):
        if len(self.xs) < 2:
            self.hold_program_counter = 0
            return 0
        input_xs = self.xs[self.press_index:self.release_index+1]
        input_ys = self.ys[self.press_index:self.release_index+1]
        input_timestamps = self.timestamps[self.press_index:self.release_index+1]

        avg = np.sum(input_ys) / len(input_xs)
        self.ys[self.press_index:self.release_index+1] = avg

        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0

    def on_smooth(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            return 0
        self.hold_program_counter = 1
    def on_smooth_clicked(self, event):
        if len(self.xs) < 2:
            self.hold_program_counter = 0
            return 0
        input_xs = self.xs[self.press_index:self.release_index+1]
        input_ys = self.ys[self.press_index:self.release_index+1]
        input_timestamps = self.timestamps[self.press_index:self.release_index+1]
        points = len(input_xs)

        order = 5
        if points <= 20:
            order = 2
        elif points <= 50:
            order = 3
        elif points <= 100:
            order = 4
        elif points <= 500:
            order = 5
        elif points <= 1000:
            order = 6
        elif points <= 2000:
            order = 7
        else:
            order = 9

        poly_zs = np.polyfit(np.asarray(input_xs), np.asarray(input_ys), order)
        poly_f = np.poly1d(poly_zs)
        poly_ys = poly_f(input_xs)
        self.ys[self.press_index:self.release_index+1] = poly_ys

        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0

    def on_load(self, event):
        pass
    def on_load_clicked(self, event):
        self.hold_program_counter = 1
        #input_filename = "output_file.csv"
        input_filename = "{}.csv".format(self.filename_textbox)
        input_directory = os.getcwd()
        input_filename_directory = "{}/{}".format(input_directory, input_filename)
        fieldnames = ["timestamps", "xs", "ys"]
        if os.path.exists(input_filename_directory):
            with open(input_filename_directory, 'r') as input_csv_file:
                self.xs = np.delete(self.xs, range(len(self.xs)))
                self.ys = np.delete(self.ys, range(len(self.ys)))
                self.timestamps = np.delete(self.timestamps, range(len(self.timestamps)))
                reader = csv.DictReader(input_csv_file)
                fieldnames = reader.fieldnames
                col_count = len(fieldnames)
                for line in reader:
                    new_dict = {}
                    for header_index in range(0, col_count):
                        new_dict[fieldnames[header_index]] = line[fieldnames[header_index]]
                    t = float(new_dict[fieldnames[0]])
                    x = float(new_dict[fieldnames[1]])
                    y = float(new_dict[fieldnames[2]])
                    self.xs = np.append(self.xs, x)
                    self.ys = np.append(self.ys, y)
                    self.timestamps = np.append(self.timestamps, t)
        else:
            print("Missing input file: {}".format(input_filename))
        self.hold_program_counter = 0

    def on_save(self, event):
        pass
    def on_save_clicked(self, event):
        output_filename = "{}.csv".format(self.filename_textbox)
        #output_filename = "output_file.csv"
        output_directory = os.getcwd()
        output_filename_directory = "{{/{}".format(output_directory, output_filename)
        fieldnames = ["timestamps", "xs", "ys"]
        if (os.path.exists(output_filename_directory)):
            os.remove(output_filename_directory)
        try:
            with open(output_filename, 'w') as output_csv_file:
                writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()
                for (x, y, t) in zip(self.xs, self.ys, self.timestamps):
                    new_dict = {"timestamps":t, "xs":x, "ys":y}
                    writer.writerow(new_dict)
        except PermissionError:
            print("ERROR: Write file caused an error. Please close the file.")

    def on_clear(self, event):
        pass
    def on_clear_clicked(self, event):
        self.xs = np.delete(self.xs, range(len(self.xs)))
        self.ys = np.delete(self.ys, range(len(self.ys)))
        self.timestamps = np.delete(self.timestamps, range(len(self.timestamps)))
        self.points = len(self.xs)
        self.index = self.points - 1

    def on_hold(self, event):
        pass
    def on_hold_clicked(self, event):
        if self.hold_program_counter:
            self.hold_program_counter = 0
        else:
            self.hold_program_counter = 1

    def on_rt(self, event):
        pass
    def on_rt_clicked(self, event):
        self.real_time_flag = 1

    def real_time_data_func(self):
        if len(self.xs) < 2:
            return 0
        self.input_xs = self.xs
        self.input_ys = self.ys
        self.input_times = self.timestamps
        self.interpolated_xs = self.input_xs
        self.linear_xs = self.input_xs
        self.poly_xs = self.input_xs
        self.__run_interpolation__()
        self.__generate_linear_equation__()
        self.__run_linear_equation__()
        self.__generate_polynomial_equation__()
        self.__run_polynomial_equation__()
        self.update_output_graph_flag = 1

    def check_graph(self, i):
        if len(self.timestamps) < 2:
            return 0
        self.update_input_graph()
        self.update_output_graph()

    def update_input_graph(self):
        if self.update_input_graph_flag:
            self.input_graph.clear()
            try:
                self.input_graph.set_ylim(min(self.ys) - 3, max(self.ys) + 3)
                #self.input_graph.set_xlim(self.times[0], self.times[-1])
            except ValueError:
                sleep(1)
                self.update_input_graph()
            self.input_graph.plot(self.timestamps, self.ys, '-', c='gray', label='original', linewidth=1.0)
            self.input_graph.plot(self.timestamps, self.ys, 'o', c='black', label='input', linewidth=1.0)
            self.input_graph.annotate("{}, {}".format(self.ys[-1], self.points), xy=[self.timestamps[-1], self.ys[-1]])
            hfmt = dates.DateFormatter('%H:%M:%S')
            self.input_graph.xaxis.set_major_formatter(hfmt)
            #self.input_graph.xaxis.set_major_locator(dates.SecondLocator())

            self.input_graph.set_title("Input Data")
            self.input_graph.set_ylabel("Price (USD)")
            self.input_graph.legend()
            plt.xticks(rotation='vertical')
            plt.subplots_adjust(bottom=.3)
        if self.press_loc:
            x_coor = self.press_loc[0]
            y_coor = self.press_loc[1]
            button_release_line = mlines.Line2D([x_coor, x_coor], [y_coor - 1000, y_coor + 1000])
            button_release_line.set_color("red")
            self.input_graph.add_line(button_release_line)
            self.input_graph.legend()
        if self.release_loc:
            x_coor = self.release_loc[0]
            y_coor = self.release_loc[1]
            button_release_line = mlines.Line2D([x_coor, x_coor], [y_coor - 1000, y_coor + 1000])
            button_release_line.set_color("red")
            self.input_graph.add_line(button_release_line)
        #if self.input_graph_overlay_position:
        #    x_point = self.input_graph_overlay_position[0]
        #    y_point = self.input_graph_overlay_position[1]
        #    self.input_graph.annotate("x={}, y={}".format(x_point, y_point), xy=[x_point, y_point])
        #    self.input_graph.plot(x_point, y_point, 'x', c='red', linewidth=1.0)
        #    print("X: {} - Y: {}".format(x_point, y_point))

    def update_output_graph(self):
        if self.update_output_graph_flag:
            self.output_graph.clear()
            self.output_graph.set_ylim(min(self.input_ys) - 3, max(self.input_ys) + 3)
            self.output_graph.set_xlim(self.input_times[0], self.input_times[-1])
            self.output_graph.plot(self.input_times, self.interpolated_ys, c="gray", label="Interpolated", linewidth=1.0)
            self.output_graph.plot(self.input_times, self.linear_ys, c="orange", label="Linear-fit", linewidth=1.0)
            if self.linear_slope:
                x_position = self.input_times[int(len(self.input_times)/2)]
                y_position = self.linear_ys[int(len(self.linear_ys) / 2)]
                self.output_graph.annotate("{}".format(round(self.linear_slope, 4)), xy=[x_position, y_position])
            self.output_graph.plot(self.input_times, self.poly_ys, c='blue', label='Poly-fit', linewidth=1.0)
            self.output_graph.set_title("Output Data")
            self.output_graph.set_xlabel("Time (Sec)")
            self.output_graph.set_ylabel("Price (USD)")

            hfmt = dates.DateFormatter('%H:%M:%S')
            self.output_graph.xaxis.set_major_formatter(hfmt)
            #self.output_graph.xaxis.set_major_locator(dates.SecondLocator())
            self.output_graph.legend()

            # Plot Avg value
            avg_value = np.sum(self.interpolated_ys) / len(self.interpolated_xs)
            avg_value = round(avg_value, 2)
            avg_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [avg_value, avg_value])
            avg_line.set_color("red")
            self.output_graph.add_line(avg_line)
            self.output_graph.annotate("{}".format(avg_value), xy=[self.input_times[0], avg_value])

            # Plot Max value
            max_value = max(self.interpolated_ys)
            max_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [max_value, max_value])
            max_line.set_color("black")
            self.output_graph.add_line(max_line)
            max_value = max(self.interpolated_ys)
            max_index = list(self.interpolated_ys).index(max_value)
            self.output_graph.annotate("{}".format(max_value), xy=[self.input_times[max_index], self.interpolated_ys[max_index]])

            # Plot Min value
            min_value = min(self.interpolated_ys)
            min_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [min_value, min_value])
            min_line.set_color("black")
            self.output_graph.add_line(min_line)
            min_value = min(self.interpolated_ys)
            min_index = list(self.interpolated_ys).index(min_value)
            self.output_graph.annotate("{}".format(min_value), xy=[self.input_times[min_index], self.interpolated_ys[min_index]])

            # Ploy-Max value
            max_value = round(max(self.poly_ys), 2)
            max_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [max_value, max_value])
            max_line.set_color("green")
            self.output_graph.add_line(max_line)
            max_value = max(self.poly_ys)
            max_index = list(self.poly_ys).index(max_value)
            self.output_graph.annotate("{}".format(round(max_value,2)), xy=[self.input_times[max_index], self.poly_ys[max_index]])

            # Ploy-Min value
            min_value = round(min(self.poly_ys), 2)
            min_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [min_value, min_value])
            min_line.set_color("green")
            self.output_graph.add_line(min_line)
            min_value = min(self.poly_ys)
            min_index = list(self.poly_ys).index(min_value)
            self.output_graph.annotate("{}".format(round(min_value, 2)), xy=[self.input_times[min_index], self.poly_ys[min_index]])


            self.update_output_graph_flag = 0


    def on_delete(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            return 0
        self.hold_program_counter = 1
    def on_delete_clicked(self, event):
        if len(self.xs) < 2:
            self.hold_program_counter = 0
            return 0
        self.xs = list(self.xs)
        del self.xs[self.press_index:self.release_index+1]
        self.ys = list(self.ys)
        del self.ys[self.press_index:self.release_index+1]
        self.timestamps = list(self.timestamps)
        del self.timestamps[self.press_index:self.release_index + 1]
        self.points = len(self.xs)
        self.index = self.points - 1
        self.xs = np.asarray(self.xs)
        self.ys = np.asarray(self.ys)
        self.timestamps = np.asarray(self.timestamps)
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0

    def button_press(self, event):
        self.hold_program_counter = 1
        self.real_time_flag = 0
        self.update_output_graph_flag = 0
        self.button_press_internel(event)
    def button_press_internel(self, event):
        if event.y < 340 or event.y > 560 or event.x > 863 or event.x < 123:
            self.hold_program_counter = 0
            return 0
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        raw_x_data = event.xdata

        if len(self.timestamps) == 0:
            self.hold_program_counter = 0
            return 0

        index = np.searchsorted(self.timestamps, [raw_x_data])[0]
        input_x_data = self.timestamps[index]
        input_y_data = self.ys[index]

        self.press_loc = (input_x_data, input_y_data)
        self.press_index = index
        return self.press_loc

    def button_release(self, event):
        if self.press_loc:
            self.button_release_internel(event)
        if self.press_loc and self.release_loc:
            self.input_xs = self.xs[self.press_index: self.release_index + 1]
            self.input_ys = self.ys[self.press_index: self.release_index + 1]
            self.input_times = self.timestamps[self.press_index: self.release_index + 1]

            self.interpolated_xs = self.input_xs
            self.linear_xs = self.input_xs
            self.poly_xs = self.input_xs
            self.__run_interpolation__()
            self.__generate_linear_equation__()
            self.__run_linear_equation__()
            self.__generate_polynomial_equation__(5)
            self.__run_polynomial_equation__()
            self.update_output_graph_flag = 1
        else:
            self.press_loc = 0
            self.release_loc = 0
            self.press_index = 0
            self.release_index = 0
        self.hold_program_counter = 0
    def button_release_internel(self, event):
        if event.y < 340 or event.y > 560 or event.x > 863 or event.x < 123:
            return 0
        if len(self.timestamps) == 0:
            return 0
        raw_x_data = event.xdata

        index = np.searchsorted(self.timestamps, [raw_x_data])[0]
        input_x_data = self.timestamps[index]
        input_y_data = self.ys[index]

        self.release_loc = (input_x_data, input_y_data)
        self.release_index = index
        if self.release_index == self.press_index:
            self.release_index = 0
            self.press_index = 0
            self.press_loc = 0
            self.release_loc = 0
            return 0
        # swap
        if self.release_index < self.press_index:
            temp = self.release_index
            self.release_index = self.press_index
            self.press_index = temp
            temp = self.release_loc
            self.release_loc = self.press_loc
            self.press_loc = temp
        return self.release_loc


    def __run_interpolation__(self):
        self.interpolated_f = interp1d(self.input_xs, self.input_ys, kind='linear')
        self.interpolated_ys = self.interpolated_f(self.interpolated_xs)
    def __round_array_list__(self, arr_list, percesion=0):
        number = 3
        if percesion:
            number = percesion
        if isinstance(arr_list, list):
            for value in arr_list:
                round(value, number)
            return arr_list
        elif isinstance(arr_list, np.ndarray):
            temp = list(arr_list)
            for value in temp:
                round(value, number)
            arr_list = np.asarray(temp)
            return arr_list
        else:
            print("list is not ndarray nor list type")
            return 1
        return 1
    def __generate_linear_equation__(self):
        self.linear_slope, self.linear_intercept, self.r_value, self.p_value, self.std_err = \
            stats.linregress(np.asarray(self.input_xs), np.asarray(self.input_ys))
    def __run_linear_equation__(self):
        self.linear_ys = (self.linear_slope * np.asarray(self.linear_xs)) + self.linear_intercept
        self.linear_ys = self.__round_array_list__(self.linear_ys, 3)
    def __generate_polynomial_equation__(self, order=0):
        if order:
            self.order = order
        else:
            if self.points <= 20:
                self.order = 2
            elif self.points <= 50:
                self.order = 3
            elif self.points <= 100:
                self.order = 4
            elif self.points <= 500:
                self.order = 5
            elif self.points <= 1000:
                self.order = 6
            elif self.points <= 2000:
                self.order = 7
            else:
                self.order = 9
        self.poly_zs = np.polyfit(np.asarray(self.input_xs), np.asarray(self.input_ys), self.order)
        self.poly_f = np.poly1d(self.poly_zs)
    def __run_polynomial_equation__(self):
        self.poly_ys = self.poly_f(self.poly_xs)



test = analyze()