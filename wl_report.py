import numpy as np
# %%

# class Report():
#     """
#     This contains methods to create reports
#     """

#     def __init__(self, filename='report', file_format='tex'):
#         self.filename = filename
#         self.format = file_format

#     def start(self):
           

#     def wl_write(self, input):
#         """
#         docstring
#         """
#         with open(f'{self.name}.{self.format}','w') as f:
#             print(input)
#             f.write(f'# {timestr}\n\n{input}')
# %%


class Report():
    def __init__(self, name = 'report', file_format = 'tex') -> None:
        self.file = f'{name}.{file_format}'
        self.timestamp()
        if file_format == 'tex':
            with open(self.file, 'a') as f:
                f.write(f'# {self.timestr}\n\n')

    def timestamp(self):
        from datetime import datetime
        self.timestr = datetime.now().strftime('%Y-%m-%d %H:%M')

    def jot(self, input):
        print(input)
        with open(self.file, 'a') as f:
            f.write(f'{input}\n\n')
            
    def erase(self):
        self.timestamp()
        with open(self.file, 'w') as f:
            f.write(f'# {self.timestr}\n\n')
# %%
