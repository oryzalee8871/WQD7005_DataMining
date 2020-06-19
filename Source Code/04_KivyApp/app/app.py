from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.clock import Clock
from plots import plot_data
from modeling import modeling, plot_loss
from crawl import get_crawl_data, check_url, check_date
from popuphelper import PopupWindow
import threading, multiprocessing,time
from functools import partial
import os
import pandas as pd
from datetime import datetime, timedelta
import queue
from functools import partial 
import matplotlib.pyplot as plt

# Create the manager
sm = ScreenManager()

def change_screen(screen_name):
    sm.current = screen_name

def work_flow(security_name, start_date, end_date): 
    security_name = security_name.strip().replace(' ', '-').lower()
    category = 'equities'
    url = f'https://www.investing.com/{category}/{security_name}-historical-data'
    if not check_url(url):
        category = 'commodities'
        url = f'https://www.investing.com/{category}/{security_name}-historical-data'
        if not check_url(url):
            PopupWindow('Invalid Security Name.', 'cancel').open()
            return
    elif not check_date(start_date):
        PopupWindow('Invalid Start Date. Please enter in "MM/DD/YYYY" format', 'cancel').open()
        return
    elif not check_date(end_date):
        PopupWindow('Invalid End Date. Please enter in "MM/DD/YYYY" format', 'cancel').open()
        return
    
    if datetime.strptime(end_date, "%m/%d/%Y") - datetime.strptime(start_date, "%m/%d/%Y") < timedelta(days=365):
        PopupWindow('Time Interval Too Short.\nPlease Select At Least One Year Interval', 'cancel').open()
        return
    
    popup = PopupWindow('Retrieving/crawling data. Please wait...', 'Abort')
    def crawl_plot():
        try:
            df = get_crawl_data(category, security_name, start_date, end_date)
            app_screen = sm.get_screen('app')
            app_screen.clear_widgets()
            app_screen.add_widget(Application(df, security_name))
            popup.dismiss()
            change_screen('app')
        except Exception as e:
            popup.dismiss()
            PopupWindow('Encountered an error.\n Please run again', 'close').open()
            raise e
    process =  threading.Thread(target=crawl_plot, daemon=True)
    process.start()
    popup.open()

class Menu(FloatLayout):
    def __init__(self, **kwargs):
        super(Menu, self).__init__(**kwargs)
        FS = 20
        top = 0.65

        self.add_widget(Image(source='logo.PNG', 
                                pos_hint={"x":0.35, "top":1},
                                size_hint= (0.3, 0.3)))

        self.add_widget(Label(text='Enter Company Name:', 
                                text_size = (230,None),
                                font_size=FS,
                                pos_hint={"x":0.2, "top":top},
                                size_hint= (0.3, 0.1)))
        security_name_input = TextInput(hint_text='sime darby plantation',
                                        multiline=False,
                                        font_size=FS,
                                        pos_hint={"x":0.5, "top":top},
                                        size_hint= (0.25, 0.1),
                                        padding_y = [FS,0])
                                    
        self.add_widget(security_name_input)

        self.add_widget(Label(text='Enter Start Date:', 
                                text_size = (230,None),
                                font_size=FS,
                                pos_hint={"x":0.2, "top":top-0.15},
                                size_hint= (0.3, 0.1)))
        start_date_input = TextInput(hint_text='MM/DD/YYYY',
                                        multiline=False,
                                        font_size=FS,
                                        pos_hint={"x":0.5, "top":top-0.15},
                                        size_hint= (0.25, 0.1),
                                        padding_y = [FS,0])
        self.add_widget(start_date_input)

        self.add_widget(Label(text='Enter End Date:', 
                                text_size = (230,None),
                                font_size=FS,
                                pos_hint={"x":0.2, "top":top-0.3},
                                size_hint= (0.3, 0.1)))
        end_date_input = TextInput(hint_text='MM/DD/YYYY',
                                    multiline=False,
                                    font_size=FS,
                                    pos_hint={"x":0.5, "top":top-0.3},
                                    size_hint= (0.25, 0.1),
                                    padding_y = [FS,0])
        self.add_widget(end_date_input)

        submit_btn = Button(text='Submit Query',
                            background_color = (62/255, 142/255, 222/255, 1),
                            font_size=FS,
                            pos_hint={"x":0.35, "top":top-0.45},
                            size_hint= (0.3, 0.1),
                            padding_y = [FS,0])  
                            
        submit_btn.bind(on_press=lambda x: work_flow(security_name_input.text, start_date_input.text, end_date_input.text))

        self.add_widget(submit_btn)
        self.background_color = (1,1,1, 1)


class Application(FloatLayout):
    def __init__(self, df, security_name, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.orientation = 'vertical'
        tab_panel= TabbedPanel()
        tab_panel.do_default_tab = False
        tab_panel.background_color = (7/255, 0, 13/255, 1)
        tab_menu = TabbedPanelItem(text="Menu")
        tab_menu.background_color = (62/255, 142/255, 222/255, 1)
        tab_chart = TabbedPanelItem(text='Chart')
        tab_chart.background_color = (62/255, 142/255, 222/255, 1)
        tab_training = TabbedPanelItem(text='Training')
        tab_training.background_color = (62/255, 142/255, 222/255, 1)
        tab_validate = TabbedPanelItem(text='Validate')
        tab_validate.background_color = (62/255, 142/255, 222/255, 1)
        tab_future = TabbedPanelItem(text='Prediction')
        tab_future.background_color = (62/255, 142/255, 222/255, 1)
        
        tab_panel.add_widget(tab_menu)
        tab_panel.add_widget(tab_chart)
        tab_panel.add_widget(tab_training)
        tab_panel.add_widget(tab_validate)
        tab_panel.add_widget(tab_future)
                            
        tab_menu.bind(on_press=lambda x: change_screen('menu'))
        
        chart_layout = FloatLayout()
        fig = plot_data(df, security_name.upper(), 30,200)
        canvas=fig.canvas
        chart_layout.add_widget(canvas)
        tab_chart.add_widget(chart_layout)

        predict_frame = FloatLayout(opacity=1)
        predict_btn = Button(text='Run Prediction',
                             background_color = (62/255, 142/255, 222/255, 1),
                             font_size=20,
                             pos_hint={"center_x":0.5, "bottom":0},
                             size_hint= (0.3, 0.075))

        predict_btn.bind(on_press=lambda x: start_predict(df, security_name))
        predict_frame.add_widget(predict_btn)
        chart_layout.add_widget(predict_frame)




        def start_predict(df, security_name):
            que = queue.Queue()
            par_modeling = partial(modeling, security_name = security_name)
            process =  threading.Thread(target=lambda q, arg1: q.put(par_modeling(arg1)), args=(que, df), daemon=True)
            process.start()

            Clock.schedule_once(lambda *args: tab_panel.switch_to(tab_training))

            
            train_fig = plt.figure(facecolor='#07000d')
            train_canvas=train_fig.canvas
            train_layout = FloatLayout()
            train_layout.add_widget(train_canvas)
            tab_training.add_widget(train_layout)


            if os.path.exists('training.csv'):
                os.remove('training.csv')
            def update_plot(fig):
                train_canvas=fig.canvas
                train_layout.clear_widgets()
                train_layout.add_widget(train_canvas)
                plt.close(fig)
                # for child in train_layout.children[:1]:
                #     train_layout.remove_widget(child)
                # Clock.schedule_once(lambda *args: tab_panel.switch_to(tab_training))

            def read_training(self):
                if os.path.exists('training.csv'):
                    loss_df = None
                    try:
                        loss_df = pd.read_csv('training.csv')
                    except Exception as e:
                        print(e)
                        pass
                    if loss_df is not None:
                        train_fig = plot_loss(loss_df)
                        update_plot(train_fig)
                if not process.is_alive():
                    Clock.unschedule(read_training)
                    val_fig, future_fig = que.get()
                    val_canvas= val_fig.canvas
                    val_layout = FloatLayout()
                    val_layout.add_widget(val_canvas)
                    tab_validate.add_widget(val_layout)
                    
                    future_canvas= future_fig.canvas
                    future_layout = FloatLayout()
                    future_layout.add_widget(future_canvas)
                    tab_future.add_widget(future_layout)

                    Clock.schedule_once(lambda *args: tab_panel.switch_to(tab_validate))
            Clock.schedule_interval(read_training, 0.1)

        
        Clock.schedule_once(lambda *args: tab_panel.switch_to(tab_chart))

        self.add_widget(tab_panel)


class PricePredictionApp(App):
    def build(self):
        menu_screen = Screen(name='menu')
        sm.add_widget(menu_screen)
        menu_screen.add_widget(Menu())
        app_screen = Screen(name='app')
        sm.add_widget(app_screen)
        return sm

if __name__ == "__main__":
    PricePredictionApp().run()