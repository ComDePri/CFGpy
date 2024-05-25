from CFGpy.behavioral import Parser


class MRIParser(Parser):
    desired_id_column_name = 'userProvidedId'   # in Parser.default_id_columns

    def __init__(self, raw_data, include_in_id: list = [], id_columns: list = Parser.default_id_columns):
        super().__init__(raw_data, include_in_id, id_columns)

    def merge_id_columns(self, data):

        data[self.id_column_name] = None  # creating new column named: self.id_column_name

        if self.desired_id_column_name in data.columns:
            data[self.id_column_name] = data[self.desired_id_column_name]

        return data


