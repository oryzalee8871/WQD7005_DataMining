
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup



def popup_window(string):
    layout = FloatLayout()
    layout.add_widget(Label(text=string, 
                            text_size=(380,None),
                            font_size=20,
                            pos_hint={"x":0.22, "top":1},
                            size_hint= (0.6, 0.2)))
    
    cancel_btn = Button(text='Cancel',
                        font_size=20,
                        pos_hint={"x":0.35, "top":0.1},
                        size_hint= (0.3, 0.1),
                        padding_y = [20,0])  
    layout.add_widget(cancel_btn)
    popupWindow = Popup(title="INFO", 
                                content=layout, 
                                size_hint=(None,None),size=(400,400),
                                auto_dismiss=False) 
    cancel_btn.bind(on_press=lambda x: popupWindow.dismiss())
    popupWindow.open()



class PopupWindow(Popup):
    def __init__(self, string, button_name, **kwargs):
        super(PopupWindow, self).__init__(**kwargs)
        layout = FloatLayout()
        layout.add_widget(Label(text=string, 
                                text_size=(380,None),
                                font_size=20,
                                pos_hint={"x":0.22, "top":0.65},
                                size_hint= (0.6, 0.2)))
        
        self.cancel_btn = Button(text=button_name,
                            background_color = (62/255, 142/255, 222/255, 1),
                            font_size=20,
                            pos_hint={"x":0.35, "top":0.15},
                            size_hint= (0.3, 0.1),
                            padding_y = [20,0])  
        layout.add_widget(self.cancel_btn)

        self.title = 'INFO'
        self.content = layout
        self.size_hint = (None,None)
        self.size = (400, 400)
        self.auto_dismiss = False
        self.cancel_btn.bind(on_press=lambda x: self.dismiss())
        