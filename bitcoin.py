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
import winsound

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
    start_val = 0
    end_val = 0

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
    alarm_task = 0
    input_graph_overlay_position = 0
    timeout_time_interval = 0
    sample_time_interval = 0

    mark1_coor = 0
    mark1_index = 0
    mark2_coor = 0
    mark2_index = 0
    alarm_coor = 0
    alarm_index = 0
    alarm_threshold = 0

    t_null = 0
    t_diff = 0
    t_end = 0
    datetime_start = 0
    datetime_start_raw = 0

    # Textbox
    filename_textbox = 0
    timeframe_textbox = 0
    sample_interval_textbox = 0
    alarm_threshold_textbox = 0

    def __init__(self):
        self.points = 0
        self.index = 0

        self.timestamps = []
        self.xs = []
        self.ys = []
        self.start_val = 0
        self.end_val = 0

        self.input_xs = []
        self.input_ys = []
        self.input_times = []

        self.count = 0
        self.timeout_time_interval = 10
        self.sample_time_interval = 10
        self.time_frame_min = 120
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

        self.mark1_coor = 0
        self.mark1_index = 0
        self.mark2_coor = 0
        self.mark2_index = 0
        self.alarm_coor = 0
        self.alarm_index = 0
        self.alarm_threshold = 10

        self.t_null = 0
        self.t_diff = 0
        self.t_end = 0
        datetime_now = datetime.datetime.now()
        self.datetime_start = datetime.datetime(datetime_now.year, datetime_now.month, datetime_now.day, 0, 0, 0, 0)
        self.datetime_start_raw = dates.date2num(self.datetime_start)
        self.filename = "CSV_FILE_{}{:02}{:02}".format(datetime_now.year,datetime_now.month,datetime_now.day)

        self.fig = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        self.input_graph = self.fig.add_subplot(2, 1, 1)
        self.output_graph = self.fig.add_subplot(2, 1, 2)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.button_press)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.button_release)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.input_graph_overlay_position_func)

        self.alarm_task = 0
        self.background_task = threading.Thread(target=self.program_loop, args=())
        self.background_task.daemon = True
        self.background_task.start()
        while(self.points < 2):
            time.sleep(3)

        self.graph_visual()

    # The graphical visual implementation for Buttons and Textfield
    def graph_visual(self):
        # Input textbox
        self.filename_textbox = "{}".format(self.filename)
        axfilebox = plt.axes([0.10, 0.20, 0.15, 0.03])
        file_text_box = TextBox(axfilebox, '', initial=self.filename_textbox)
        file_text_box.on_submit(self.file_submit)
        # Save button
        axSave = plt.axes([0.10, 0.16, 0.15, 0.03])
        bSave = Button(axSave, 'Save')
        bSave.on_clicked(self.on_save_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_save)
        # Load button
        axLoad = plt.axes([0.10, 0.12, 0.15, 0.03])
        bLoad = Button(axLoad, 'Load')
        bLoad.on_clicked(self.on_load_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_load)

        # Average button
        axAverage = plt.axes([0.26, 0.20, 0.15, 0.03])
        bAverage = Button(axAverage, 'Average (selection)')
        bAverage.on_clicked(self.on_average_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_average)
        # Smooth button
        axSmooth = plt.axes([0.26, 0.16, 0.15, 0.03])
        bSmooth = Button(axSmooth, 'Smooth (selection)')
        bSmooth.on_clicked(self.on_smooth_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_smooth)
        # Max button
        axMax = plt.axes([0.26, 0.12, 0.15, 0.03])
        bMax = Button(axMax, 'Max (selection)')
        bMax.on_clicked(self.on_max_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_max)
        # Min button
        axMin = plt.axes([0.26, 0.08, 0.15, 0.03])
        bMin = Button(axMin, 'Min (selection)')
        bMin.on_clicked(self.on_min_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_min)
        # Linearize button
        axLinearize = plt.axes([0.26, 0.04, 0.15, 0.03])
        bLinearize = Button(axLinearize, 'Linearize (selection)')
        bLinearize.on_clicked(self.on_linearize_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_linearize)

        # Live button
        axRT = plt.axes([0.42, 0.20, 0.15, 0.03])
        bRT = Button(axRT, 'Realtime')
        bRT.on_clicked(self.on_rt_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_rt)
        # Clear button
        axClear = plt.axes([0.42, 0.16, 0.15, 0.03])
        bClear = Button(axClear, 'Clear Data')
        bClear.on_clicked(self.on_clear_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_clear)
        # Delete button
        axDelete = plt.axes([0.42, 0.12, 0.15, 0.03])
        bDelete = Button(axDelete, 'Delete (selection)')
        bDelete.on_clicked(self.on_delete_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_delete)
        # Select All button
        axSelectAll = plt.axes([0.42, 0.08, 0.15, 0.03])
        bSelectAll = Button(axSelectAll, 'Select All')
        bSelectAll.on_clicked(self.on_select_all_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_select_all)
        # Reconstruct button
        axReconstruct = plt.axes([0.42, 0.04, 0.15, 0.03])
        bReconstruct = Button(axReconstruct, 'Reconstruct (selection)')
        bReconstruct.on_clicked(self.on_reconstruct_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_reconstruct)

        # Timeframe textbox
        self.timeframe_textbox = "{}".format(self.time_frame_min)
        axtimeframebox = plt.axes([0.90, 0.20, 0.05, 0.03])
        time_frame_text_box = TextBox(axtimeframebox, 'Span (min):', initial=self.timeframe_textbox)
        time_frame_text_box.on_submit(self.time_frame_submit)
        # Sample Interval textbox
        self.sample_interval_textbox = "{}".format(self.sample_time_interval)
        axsampleintervalbox = plt.axes([0.90, 0.16, 0.05, 0.03])
        sample_interval_text_box = TextBox(axsampleintervalbox, 'Interval:', initial=self.sample_interval_textbox)
        sample_interval_text_box.on_submit(self.sample_interval_submit)
        # Hold button
        axHold = plt.axes([0.80, 0.12, 0.15, 0.03])
        bHold = Button(axHold, 'Hold')
        bHold.on_clicked(self.on_hold_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_hold)
        # Close button
        axClose = plt.axes([0.80, 0.08, 0.15, 0.03])
        bClose = Button(axClose, 'Close')
        bClose.on_clicked(self.on_close_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_close)
        # Mark 1 button
        axMark1 = plt.axes([0.58, 0.20, 0.15, 0.03])
        bMark1 = Button(axMark1, 'Mark 1')
        bMark1.on_clicked(self.on_mark1_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mark1)
        # Mark 2 button
        axMark2 = plt.axes([0.58, 0.16, 0.15, 0.03])
        bMark2 = Button(axMark2, 'Mark 2')
        bMark2.on_clicked(self.on_mark2_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mark2)
        # Alarm button
        axAlarm = plt.axes([0.58, 0.12, 0.15, 0.03])
        bAlarm = Button(axAlarm, 'Alarm')
        bAlarm.on_clicked(self.on_alarm_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_alarm)
        # Alarm threshold textbox
        self.alarm_threshold_textbox = "{}".format(self.alarm_threshold)
        ax_alarm_threshold_textbox = plt.axes([0.74, 0.12, 0.03, 0.03])
        alarm_threshold_text_box = TextBox(ax_alarm_threshold_textbox, '', initial=self.alarm_threshold_textbox)
        alarm_threshold_text_box.on_submit(self.alarm_threshold_submit)
        plt.text(.01,1.3,"Limit",fontsize=8,color='black')

        # Clear Marker button
        axClearMarker = plt.axes([0.58, 0.08, 0.15, 0.03])
        bClearMarker = Button(axClearMarker, 'Clear Marker')
        bClearMarker.on_clicked(self.on_clear_marker_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self.on_clear_marker)

        ani = animation.FuncAnimation(self.fig, self.check_graph, interval=100)

        plt.xticks(rotation='vertical')
        plt.subplots_adjust(bottom=.3)
        plt.show()
    # The main program function that retrieve data and interpret data as background task
    def program_loop(self):
        self.t_null = time.time()
        while (not self.close_program_flag):
            t_new = time.time()
            if not self.hold_program_counter:
                self.new_data()
            if not self.hold_program_counter:
                if self.real_time_flag:
                    self.real_time_data_func()
            sleep(self.sample_time_interval)
            self.t_end = time.time()
            self.t_diff = round(self.t_end - t_new, 2)
            t_start = round(self.t_end - self.t_null, 2)
            print("{}/{}. t_start: {} t_diff: {} time_frame: {}s".format(self.points, self.count, t_start, self.t_diff, self.time_frame_min*60))
    # This block stores new data to timestamps, xs, ys, and update some of the information variables
    def new_data(self):
        self.update_input_graph_flag = 0
        if self.points == self.count:
            self.points = self.points - 1
            self.index = self.index - 1
            self.timestamps = np.delete(self.timestamps, 0)
            self.xs = np.delete(self.xs, 0)
            self.ys = np.delete(self.ys, 0)

        now = datetime.datetime.now()
        date = dates.date2num(now)
        x = date
        y = self.__bitcoin_order_book__()
        print("\n=== {} === {:02}:{:02}:{:02} ===".format(y, now.hour, now.minute, now.second))
        if y == -1:
            print("Value readout error! Trying again.")
            while y == -1:
                sleep(self.timeout_time_interval)
                y = self.__bitcoin_order_book__()
                print("\n=== {} === {:02}:{:02}:{:02} ===".format(y, now.hour, now.minute, now.second))

        self.ys = np.append(self.ys, y)
        self.timestamps = np.append(self.timestamps, x)
        self.xs = np.append(self.xs, len(self.timestamps))
        self.points = len(self.timestamps)
        self.index = self.points - 1
        self.start_val = (self.timestamps[0], round(self.ys[0], 2))
        self.end_val = (self.timestamps[-1], round(self.ys[-1], 2))
        self.update_input_graph_flag = 1
    # Retrieve raw data that is specific to gdax and coinbase
    def __bitcoin_order_book__(self):
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
    # Handler for sample interval textbox

    # Function runs at different thread to update the graph
    def check_graph(self, i):
        if len(self.timestamps) < 2:
            return 0
        self.update_input_graph()
        self.update_output_graph()
    # Updates the input graph at each call
    def update_input_graph(self):
        if self.update_input_graph_flag:
            self.input_graph.clear()
            try:
                self.input_graph.set_ylim(min(self.ys) - 12, max(self.ys) + 12)
                self.input_graph.set_xlim(self.timestamps[0], self.timestamps[-1])
            except ValueError:
                print("Value error: update input graph")
                exit(1)
            self.input_graph.plot(self.timestamps, self.ys, '-', c='gray', label='original', linewidth=1.0)
            self.input_graph.plot(self.timestamps, self.ys, 'o', c='black', label='input', linewidth=1.0)

            self.input_graph.annotate("{}".format(int(self.ys[-1])), xy=[self.timestamps[-1], self.ys[-1]+1])
            hfmt = dates.DateFormatter('%H:%M:%S')
            self.input_graph.xaxis.set_major_formatter(hfmt)

            self.input_graph.text(x=self.timestamps[0], y=max(self.ys)+15, text="Start: {}".format(self.start_val[1]), s="", color='black')
            self.input_graph.text(x=self.timestamps[-1], y=max(self.ys)+15, text="End: {}".format(self.end_val[1]), s="", color='black')

            self.input_graph.set_title("Input Data")
            self.input_graph.set_ylabel("Value")
            plt.xticks(rotation='vertical')
            plt.subplots_adjust(bottom=.3)
            if self.press_loc:
                x_coor = self.press_loc[0]
                y_coor = self.press_loc[1]
                self.input_graph.annotate("{}".format(round(y_coor, 3)), xy=[x_coor, max(self.ys)+6], color='firebrick')
                self.input_graph.plot(self.press_loc[0], self.press_loc[1], 'o', c='firebrick', label='press', linewidth=1.0)
            if self.release_loc:
                x_coor = self.release_loc[0]
                y_coor = self.release_loc[1]
                self.input_graph.annotate("{}".format(round(y_coor, 3)), xy=[x_coor, max(self.ys)+9], color='darkred')
                self.input_graph.plot(self.release_loc[0], self.release_loc[1], 'o', c='darkred', label='release', linewidth=1.0)
            if self.mark1_coor:
                x_coor = self.mark1_coor[0]
                y_coor = self.mark1_coor[1]
                x_coor_str = self.print_raw_format(x_coor)
                self.input_graph.annotate("{}, {}".format(x_coor_str, round(y_coor, 3)), xy=[x_coor, min(self.ys)-3], color='darkblue')
                self.input_graph.plot(self.mark1_coor[0], self.mark1_coor[1], 'o', c='darkblue', label='mark1', linewidth=1.0)
            if self.mark2_coor:
                x_coor = self.mark2_coor[0]
                y_coor = self.mark2_coor[1]
                x_coor_str = self.print_raw_format(x_coor)
                self.input_graph.annotate("{}, {}".format(x_coor_str, round(y_coor, 3)), xy=[x_coor, min(self.ys)-6], color='darkgreen')
                self.input_graph.plot(self.mark2_coor[0], self.mark2_coor[1], 'o', c='darkgreen', label='mark2', linewidth=1.0)
            if self.alarm_coor:
                x_coor = self.alarm_coor[0]
                y_coor = self.alarm_coor[1]
                button_release_line = mlines.Line2D([x_coor, x_coor], [y_coor - self.alarm_threshold, y_coor + self.alarm_threshold], color='darkgoldenrod')
                self.input_graph.add_line(button_release_line)
                x_coor_str = self.print_raw_format(x_coor)
                self.input_graph.annotate("{}, {}".format(x_coor_str, round(y_coor, 3)), xy=[x_coor, min(self.ys)-9], color='darkgoldenrod')
                self.input_graph.plot(self.alarm_coor[0], self.alarm_coor[1], 'o', c='darkgoldenrod', label='alarm', linewidth=1.0)

            if self.input_graph_overlay_position:
                x_coor = self.input_graph_overlay_position[0]
                x_coor_str = self.print_raw_format(x_coor)
                y_coor = self.input_graph_overlay_position[1]
                self.input_graph.annotate("x={}, y={}".format(x_coor_str, y_coor), xy=[x_coor, y_coor+3])
                self.input_graph.plot(x_coor, y_coor, 'x', c='red', linewidth=1.0)

        self.update_input_graph_flag = 0
    # Function that update output graph at each call
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
            self.output_graph.set_ylabel("Value")

            hfmt = dates.DateFormatter('%H:%M:%S')
            self.output_graph.xaxis.set_major_formatter(hfmt)
            #self.output_graph.xaxis.set_major_locator(dates.SecondLocator())
            #self.output_graph.legend()

            # Plot Avg value
            avg_value = np.sum(self.interpolated_ys) / len(self.interpolated_xs)
            avg_value = round(avg_value, 2)
            avg_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [avg_value, avg_value])
            avg_line.set_color("darkred")
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
            max_line.set_color("darkgreen")
            self.output_graph.add_line(max_line)
            max_value = max(self.poly_ys)
            max_index = list(self.poly_ys).index(max_value)
            self.output_graph.annotate("{}".format(round(max_value,2)), xy=[self.input_times[max_index], self.poly_ys[max_index]])

            # Ploy-Min value
            min_value = round(min(self.poly_ys), 2)
            min_line = mlines.Line2D([self.input_times[0], self.input_times[-1]], [min_value, min_value])
            min_line.set_color("darkgreen")
            self.output_graph.add_line(min_line)
            min_value = min(self.poly_ys)
            min_index = list(self.poly_ys).index(min_value)
            self.output_graph.annotate("{}".format(round(min_value, 2)), xy=[self.input_times[min_index], self.poly_ys[min_index]])

            self.update_output_graph_flag = 0
    # Handler for updating input graph overlay position
    def input_graph_overlay_position_func(self, event):
        if event.x > 860:
            return 0
        elif event.x < 130:
            return 0
        elif event.y > 560:
            return 0
        elif event.y < 400:
            return 0
        self.input_graph_overlay_position = 0
        raw_x_data = event.xdata
        raw_y_data = event.ydata
        try:
            index = np.searchsorted(self.timestamps, [raw_x_data])[0]
            input_x_data = self.timestamps[index]
        except IndexError:
            return 0
        input_y_data = self.ys[index]
        self.input_graph_overlay_position = (input_x_data, input_y_data)
        self.update_input_graph_flag = 1
    # Handler for sample interval
    def sample_interval_submit(self, text):
        initial = self.sample_time_interval
        try:
            self.sample_time_interval = int("{}".format(text))
        except ValueError:
            print("Value Error")
            self.sample_time_interval = initial
            return 0
        self.__set_count__()
    # Handler for time frame textbox.
    def time_frame_submit(self, text):
        initial = self.time_frame_min
        try:
            self.time_frame_min = int("{}".format(text))
        except ValueError:
            print("Value Error")
            self.time_frame_min = initial
            return 0
        self.__set_count__()
    # Based on a derived formula, this function determines count variable for input time frame and sample time interval
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
    # Handler for Mark 1 component upon click
    def on_mark1(self, event):
        pass
    def on_mark1_clicked(self, event):
        if self.mark1_coor:
            self.mark1_coor = 0
            self.mark1_index = 0
        elif self.press_loc:
            self.mark1_coor = self.press_loc
            self.mark1_index = self.press_index
            self.press_loc = 0
            self.press_index = 0
        self.update_input_graph_flag = 1
    # Handler for Mark 2 component upon click
    def on_mark2(self, event):
        pass
    def on_mark2_clicked(self, event):
        if self.mark2_coor:
            self.mark2_coor = 0
            self.mark2_index = 0
        elif self.press_loc:
            self.mark2_coor = self.press_loc
            self.mark2_index = self.press_index = 0
            self.press_loc = 0
            self.press_index = 0
        self.update_input_graph_flag = 1
    # Handler for Clearing all the markers
    def on_clear_marker(self, event):
        pass
    def on_clear_marker_clicked(self, event):
        self.press_index = 0
        self.press_loc = 0
        self.release_index = 0
        self.release_loc = 0
        self.mark1_index = 0
        self.mark1_coor = 0
        self.mark2_index = 0
        self.mark2_coor = 0
        self.alarm_coor = 0
        self.alarm_index = 0
        self.update_input_graph_flag = 1
    # Fix the marker coordinate position when shrinking or extending the data size
    def fix_marker_coor(self):
        if self.mark1_coor:
            y_val = self.mark1_coor[1]
            index = self.ys.index(y_val)
            x_val = self.timestamps[index]
            self.mark1_index = index
            self.mark1_coor = (x_val, y_val)
        if self.mark2_coor:
            y_val = self.mark2_coor[1]
            index = self.ys.index(y_val)
            x_val = self.timestamps[index]
            self.mark2_index = index
            self.mark2_coor = (x_val, y_val)
        if self.alarm_coor:
            y_val = self.alarm_coor[1]
            index = self.ys.index(y_val)
            x_val = self.timestamps[index]
            self.alarm_index = index
            self.alarm_coor = (x_val, y_val)
        self.update_input_graph_flag = 1
    # Separate thread that runs in the background that check for alarm value once set
    def check_alarm(self):
        print("Alarm will sound when {} above or below {}".format(self.alarm_threshold, self.alarm_coor[1]))
        max_value = self.alarm_coor[1] + self.alarm_threshold
        min_value = self.alarm_coor[1] - self.alarm_threshold
        print("Checking alarm for in range {}-{}".format(min_value, max_value))
        while(self.alarm_coor):
            max_value = self.alarm_coor[1] + self.alarm_threshold
            min_value = self.alarm_coor[1] - self.alarm_threshold
            if self.ys[-1] > min_value and self.ys[-1] < max_value:
                duration = 1000
                freq = 440
                winsound.Beep(freq, duration)
            time.sleep((self.sample_time_interval)/2)
        self.alarm_task = 0
        print("Alarm deactivated")
    # Handler for threshold textbox that determines at what range the alarm should sound
    def alarm_threshold_submit(self, text):
        initial = self.alarm_threshold
        try:
            self.alarm_threshold = int("{}".format(text))
        except ValueError:
            print("Value Error")
            self.alarm_threshold = initial
            return 0
    def on_alarm(self, event):
        pass
    # Handler for alarm button, thus fires off alarm thread in the background to check for alarm set value
    def on_alarm_clicked(self, event):
        if self.alarm_coor:
            print("Deactivating alarm set at {} for {}".format(self.alarm_coor[0], self.alarm_coor[1]))
            self.alarm_coor = 0
            self.alarm_index = 0
        elif self.press_loc and self.alarm_task == 0:
            self.alarm_coor = self.press_loc
            print("Set alarm at {} for {}".format(self.alarm_coor[0], self.alarm_coor[1]))
            self.alarm_index = self.press_index
            self.press_loc = 0
            self.press_index = 0
            self.alarm_task = threading.Thread(target=self.check_alarm, args=())
            self.alarm_task.daemon = True
            self.alarm_task.start()
        self.update_input_graph_flag = 1
    # Handler for calculating average value
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
        avg = round(avg, 3)
        self.ys[self.press_index:self.release_index+1] = avg

        self.fix_marker_coor()
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
    # Handler for calculating max value and rewrite input data
    def on_max(self, event):
        pass
    def on_max_clicked(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            self.hold_program_counter = 0
            return 0
        self.hold_program_counter = 1

        input_xs = self.xs[self.press_index:self.release_index + 1]
        input_ys = self.ys[self.press_index:self.release_index + 1]
        input_timestamps = self.timestamps[self.press_index:self.release_index + 1]

        max = np.max(input_ys)
        self.ys[self.press_index:self.release_index + 1] = max

        self.fix_marker_coor()
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
    # Handler for calculating min value and rewrite input data
    def on_min(self, event):
        pass
    def on_min_clicked(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            self.hold_program_counter = 0
            return 0
        self.hold_program_counter = 1

        input_xs = self.xs[self.press_index:self.release_index + 1]
        input_ys = self.ys[self.press_index:self.release_index + 1]
        input_timestamps = self.timestamps[self.press_index:self.release_index + 1]

        min = np.min(input_ys)
        self.ys[self.press_index:self.release_index + 1] = min

        self.fix_marker_coor()
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
    # Handler for calculating linear function and updating input data
    def on_linearize(self, event):
        pass
    def on_linearize_clicked(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            self.hold_program_counter = 0
            return 0
        self.hold_program_counter = 1

        input_xs = self.xs[self.press_index:self.release_index + 1]
        input_ys = self.ys[self.press_index:self.release_index + 1]
        input_timestamps = self.timestamps[self.press_index:self.release_index + 1]

        linear_slope, linear_intercept, r_value, p_value, std_err = \
            stats.linregress(np.asarray(input_xs), np.asarray(input_ys))
        linear_ys = (linear_slope * np.asarray(input_xs)) + linear_intercept
        linear_ys = self.__round_array_list__(linear_ys, 3)
        self.ys[self.press_index:self.release_index + 1] = linear_ys

        self.fix_marker_coor()
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
    # Handler for calculating polynomial function and updating input data
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
        poly_ys = self.__round_array_list__(poly_ys, 3)
        self.ys[self.press_index:self.release_index+1] = poly_ys

        self.fix_marker_coor()
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
    # Handler for calculating and reconstructing input data for data loss
    def on_reconstruct(self, event):
        pass
    def on_reconstruct_clicked(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            self.hold_program_counter = 0
            return 0
        self.hold_program_counter = 1
        self.update_input_graph_flag = 0

        press_index = 0
        release_index = len(self.timestamps) - 1
        input_xs = self.xs[press_index:release_index + 1]
        input_ys = self.ys[press_index:release_index + 1]

        order = 5
        if self.points <= 20:
            order = 2
        elif self.points <= 50:
            order = 3
        elif self.points <= 100:
            order = 4
        elif self.points <= 500:
            order = 5
        elif self.points <= 1000:
            order = 6
        elif self.points <= 2000:
            order = 7
        else:
            order = 9
        poly_zs = np.polyfit(np.asarray(input_xs), np.asarray(input_ys), order)
        poly_f = np.poly1d(poly_zs)

        timestamp_fix = self.timestamps[self.press_index: self.release_index + 1]
        if len(timestamp_fix) != 2:
            print("Requirement: Pick adjacent node to rebuild inner data")
            self.press_loc = 0
            self.release_loc = 0
            self.press_index = 0
            self.release_index = 0
            self.hold_program_counter = 0
            self.update_input_graph_flag = 1
            return 0

        timestamp_fix = self.adjust_time_axes(timestamp_fix[0], timestamp_fix[-1], self.sample_time_interval)
        timestamp_list = list(self.timestamps)
        for index in range(len(timestamp_fix) - 2):
            timestamp_list.insert(self.press_index + index + 1, timestamp_fix[index + 1])
        self.timestamps = np.asarray(timestamp_list)
        xs = np.linspace(1, len(timestamp_list), len(timestamp_list))
        self.xs = np.asarray(xs)

        ys_fix_list = list(self.ys)
        for index in range(len(timestamp_fix) - 2):
            y = poly_f(self.xs[self.press_index + index])
            y = round(y, 2)
            ys_fix_list.insert(self.press_index + index + 1, y)
        self.ys = np.asarray(ys_fix_list)

        self.points = len(self.timestamps)
        self.index = self.points - 1

        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
        self.update_input_graph_flag = 1
    # Handler for updating filename information
    def file_submit(self, text):
        self.filename_textbox = "{}.csv".format(text)
    # Handler for load button once clicked to load specified csv file
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
    # Handler for save button once clicked to store data into specified csv file
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
    # Handler to clear data input data
    def on_clear(self, event):
        pass
    def on_clear_clicked(self, event):
        self.hold_program_counter = 1
        self.xs = np.delete(self.xs, range(len(self.xs)))
        self.ys = np.delete(self.ys, range(len(self.ys)))
        self.timestamps = np.delete(self.timestamps, range(len(self.timestamps)))
        self.points = len(self.xs)
        self.index = self.points - 1
        self.on_delete_clicked(event)
        self.update_input_graph_flag = 1
    # Handler to close the program
    def on_close(self, event):
        pass
    def on_close_clicked(self, event):
        self.close_program_flag = 1
    # Handler to pause the pause
    def on_hold(self, event):
        pass
    def on_hold_clicked(self, event):
        if self.hold_program_counter:
            self.hold_program_counter = 0
        else:
            self.hold_program_counter = 1
    # Handler to process the input data in real-time as the input data updates
    def on_rt(self, event):
        pass
    def on_rt_clicked(self, event):
        self.press_loc = 0
        self.press_index = 0
        self.release_loc = 0
        self.release_index = 0
        self.real_time_flag = 1
    # This function gets called at each cycle when real time flag set
    def real_time_data_func(self):
        if len(self.xs) < 2:
            return 0
        if self.press_loc:
            self.release_index = len(self.timestamps)-1
            self.input_xs = self.xs[self.press_index: self.release_index + 1]
            self.input_ys = self.ys[self.press_index: self.release_index + 1]
            self.input_times = self.timestamps[self.press_index: self.release_index + 1]
            self.interpolated_xs = self.input_xs
            self.linear_xs = self.input_xs
            self.poly_xs = self.input_xs
        else:
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
    # Handler for deleting data for selected region
    def on_delete(self, event):
        if len(self.timestamps) < 2:
            self.hold_program_counter = 0
            return 0
        if not (self.press_loc and self.release_loc):
            return 0
        self.hold_program_counter = 1
    def on_delete_clicked(self, event):
        if not len(self.xs):
            self.hold_program_counter = 0
            return 0
        self.ys = list(self.ys)
        del self.ys[self.press_index:self.release_index+1]
        self.ys = np.asarray(self.ys)

        self.timestamps = list(self.timestamps)
        del self.timestamps[self.press_index:self.release_index + 1]
        self.timestamps = np.asarray(self.timestamps)

        self.points = len(self.timestamps)
        self.index = self.points - 1
        self.xs = np.linspace(1, self.points, self.points)

        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        self.hold_program_counter = 0
        self.update_input_graph_flag = 1
    # Handler to select all data for selected region
    def on_select_all(self, event):
        pass
    def on_select_all_clicked(self, event):
        self.update_output_graph_flag = 0
        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0

        index = 0
        input_x_data = self.timestamps[index]
        input_y_data = self.ys[index]

        self.press_loc = (input_x_data, input_y_data)
        self.press_index = index

        index = len(self.timestamps)-1
        input_x_data = self.timestamps[index]
        input_y_data = self.ys[index]

        self.release_loc = (input_x_data, input_y_data)
        self.release_index = index

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
        self.update_input_graph_flag = 1
        self.update_output_graph_flag = 1
    # Handler to process the mouse click press button on input graph
    def button_press(self, event):
        self.hold_program_counter = 1
        self.real_time_flag = 0
        self.update_output_graph_flag = 0
        self.update_input_graph_flag = 1
        self.button_press_internel(event)
    def button_press_internel(self, event):
        if event.x > 860:
            return 0
        elif event.x < 130:
            return 0
        elif event.y > 560:
            return 0
        elif event.y < 400:
            return 0

        self.press_loc = 0
        self.release_loc = 0
        self.press_index = 0
        self.release_index = 0
        raw_x_data = event.xdata

        if len(self.timestamps) == 0:
            self.hold_program_counter = 0
            return 0
        try:
            index = np.searchsorted(self.timestamps, [raw_x_data])[0]
            input_x_data = self.timestamps[index]
        except IndexError:
            return 0
        input_y_data = self.ys[index]

        self.press_loc = (input_x_data, input_y_data)
        self.press_index = index
        return self.press_loc
    # Handler to process the mouse click release button on input graph
    def button_release(self, event):
        if self.press_loc:
            self.button_release_internel(event)
        self.input_xs = self.xs[self.press_index: self.release_index + 1]
        self.input_ys = self.ys[self.press_index: self.release_index + 1]
        self.input_times = self.timestamps[self.press_index: self.release_index + 1]
        if len(self.input_xs) < 2:
            self.hold_program_counter = 0
            return 0

        if self.press_loc and self.release_loc:
            self.interpolated_xs = self.input_xs
            self.linear_xs = self.input_xs
            self.poly_xs = self.input_xs
            self.__run_interpolation__()
            self.__generate_linear_equation__()
            self.__run_linear_equation__()
            self.__generate_polynomial_equation__(5)
            self.__run_polynomial_equation__()
            self.update_output_graph_flag = 1
        elif self.press_loc:
            self.interpolated_xs = self.input_xs
            self.linear_xs = self.input_xs
            self.poly_xs = self.input_xs
            self.__run_interpolation__()
            self.__generate_linear_equation__()
            self.__run_linear_equation__()
            self.__generate_polynomial_equation__(5)
            self.__run_polynomial_equation__()
            self.update_output_graph_flag = 1
            self.real_time_flag = 1
        else:
            self.press_loc = 0
            self.release_loc = 0
            self.press_index = 0
            self.release_index = 0
        self.hold_program_counter = 0
    def button_release_internel(self, event):
        if event.x > 860:
            return 0
        elif event.x < 130:
            return 0
        elif event.y > 560:
            return 0
        elif event.y < 400:
            return 0
        raw_x_data = event.xdata
        try:
            index = np.searchsorted(self.timestamps, [raw_x_data])[0]
            input_x_data = self.timestamps[index]
        except IndexError:
            return 0
        input_y_data = self.ys[index]

        self.release_loc = (input_x_data, input_y_data)
        self.release_index = index
        if self.release_index == self.press_index:
            self.release_index = len(self.timestamps)-1
            #self.press_index = 0
            #self.press_loc = 0
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
        self.update_input_graph_flag = 1
        return self.release_loc

    # Generate interpolated function and output interpolated data
    def __run_interpolation__(self):
        self.interpolated_f = interp1d(self.input_xs, self.input_ys, kind='linear')
        self.interpolated_ys = self.interpolated_f(self.interpolated_xs)
    # Round array list to a decimal number
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
    # Generate linear equation for input data
    def __generate_linear_equation__(self):
        self.linear_slope, self.linear_intercept, self.r_value, self.p_value, self.std_err = \
            stats.linregress(np.asarray(self.input_xs), np.asarray(self.input_ys))
    # Run linear equation on linear_xs and output data to linear_ys
    def __run_linear_equation__(self):
        self.linear_ys = (self.linear_slope * np.asarray(self.linear_xs)) + self.linear_intercept
        self.linear_ys = self.__round_array_list__(self.linear_ys, 3)
    # Generate polynomial equation from input xs and ys
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
    # Run polynomial equation on poly xs and output to poly ys
    def __run_polynomial_equation__(self):
        self.poly_ys = self.poly_f(self.poly_xs)

    # Time axes formatting and conversion
    def seconds_to_raw(self, seconds):
        datetime_seconds_raw = self.datetime_start + datetime.timedelta(seconds=seconds)
        datetime_seconds_raw = dates.date2num(datetime_seconds_raw)
        datetime_diff_raw = datetime_seconds_raw - self.datetime_start_raw
        return datetime_diff_raw
    def raw_to_datetime(self, raw):
        datetime_fmt = dates.num2timedelta(raw) + self.datetime_start
        return datetime_fmt
    def print_raw_format(self, raw):
        datetime_diff_raw = self.datetime_start_raw + raw
        datetime_diff = dates.num2date(datetime_diff_raw)
        hfmt = "%H:%M:%S"
        time_str = datetime_diff.strftime(hfmt)
        return time_str
    def print_datetime_format(self, date_time):
        hfmt = "%H:%M:%S"
        time_str = date_time.strftime(hfmt)
        return time_str
    def time_to_second(self, time_str):
        time_arr = time_str.split(':')
        hr = int(time_arr[0])
        min = int(time_arr[1])
        sec = int(time_arr[2])
        seconds = (hr * 3600) + (min * 60) + sec
        return seconds
    def time_to_datetime(self, time_str):
        seconds = self.time_to_second(time_str)
        num = self.seconds_to_raw(seconds)
        date_time = dates.num2timedelta(num) + self.datetime_start
        return date_time
    def datetime_to_seconds(self, date_time):
        hr = date_time.hour
        min = date_time.minute
        sec = date_time.second
        return ((hr * 3600) + (min * 60) + sec)
    def raw_to_seconds(self, raw):
        date_time = self.raw_to_datetime(raw)
        seconds = self.datetime_to_seconds(date_time)
        return seconds
    def datetime_advance(self, date_time, sec):
        seconds = self.datetime_to_seconds(date_time)
        seconds = seconds + sec
        raw = self.seconds_to_raw(seconds)
        return self.raw_to_datetime(raw)
    def raw_advance(self, raw, sec):
        seconds = self.raw_to_seconds(raw)
        seconds = sec + seconds
        raw = self.seconds_to_raw(seconds)
        date_time = self.raw_to_datetime(raw)
        return dates.date2num(date_time)
    def datetime_to_raw(self, date_time):
        new_raw = dates.date2num(date_time)
        return new_raw
    def adjust_time_axes(self, t_start_raw, t_end_raw, timeout=0):
        time_out = 10
        if timeout:
            time_out = timeout
        t_end_start_diff = t_end_raw - t_start_raw
        t_end_start_diff_sec = self.raw_to_seconds(t_end_start_diff)
        count = int(round(t_end_start_diff_sec / time_out, 0))
        time_axes = []
        time_axes.append(t_start_raw)
        for index in range(count):
            last_val = time_axes[-1]
            last_datetime = self.raw_to_datetime(last_val)
            new_datetime = self.datetime_advance(last_datetime, time_out)
            new_raw = self.datetime_to_raw(new_datetime)
            time_axes.append(new_raw)
        return time_axes


test = analyze()