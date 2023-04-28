'''
Sports2D
=========================

Compute joint angles from a video.
The movement needs to be filmed from the side (in the sagittal plane).

Be aware of the limitations:
- 

'''

from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivymd.app import MDApp


class Sports2DScreen(Screen):
    def add_widget(self, *args, **kwargs):
        return super(Sports2DScreen, self).add_widget(*args, **kwargs)

class Sports2DApp(MDApp):
    def build(self):
        self.icon = 'Data/cursor.png'
        self.theme_cls.theme_style = "Dark"


if __name__ == '__main__':
    Sports2DApp().run()

