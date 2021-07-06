#
#  batch_analysis.py
#  OpenSALTO
#
#  An OpenSALTO batch analysis report plugin
#
#  Copyright 2016–2021 Mediavec Ky. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto
import os, math
from datetime import datetime
import csv, xlsxwriter
from itertools import chain

class ColumnWidths():
    def __init__(self, data):
        self.widths = [8.0] * len(data)
        self.update(data)
    def update(self, data):
        for i in range(len(data)):
            if type(data[i]) == datetime and self.widths[i] < 16:
                self.widths[i] = 16
            elif type(data[i]) == str and len(data[i]) > self.widths[i]:
                self.widths[i] = len(data[i])

class Plugin(salto.Plugin):
    """OpenSALTO plugin for batch analysis
        
        Expects the analysis plugin to return a report in the result dictionary
        with key html (for html reports), or table (for table data). Table
        reports should be dictionaries with keys header,  data, and summary for
        a header row, data rows, and summary rows, respectively. The summary is
        ignored if a flat table format is specified."""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        inputs = [('channelTable', 'S', 0, 0),
                  ('source', 'O', "list of data files or directories", None),
                  ('source_format', 'S', "name of the file format plugin to be used", 'directory'),
                  ('result_file_path', 'S', "path of the result file", None),
                  ('result_format', 'S', "format of the result file", 'xlsx'),
                  ('analysis', 'S', "name of the analysis plugin to be used", None),
                  ('parameters', 'f', "input parameters for the analysis plugin", {}),
                  ('unravel_keys', 'O', "sequence of parameter keys that should be unraveled", ()),
                  ('options', 'O', "sequence of options ('flat', 'structured')", ('flat')),
                  ('communicator', 'O', "MPI communicator object (optional)", None)
                  ]
        self.registerComputation('batch analysis', self._report,
                                 inputs = inputs,
                                 outputs = [])

    @staticmethod
    def _mergeHeaders(headers):
        """Merges a list of headers that have a common order but may have
           missing items
        """
        longest = max(*headers, key=len)
        full = list(set(chain(*headers)))
        result = longest + [key for key in full if key not in longest]
        for h in headers:
            while True:
                order = [result.index(item) for item in h]
                swap = [(o, s) for o, s in zip(order, sorted(order)) if o != s]
                if swap:
                    result.insert(swap[0][1], result.pop(swap[0][0]))
                else:
                    break
        return result

    @staticmethod
    def _columnOrder(source, target):
        """Returns the source indices for items in target or -1 if not found"""
        def index(item):
            if item in source:
                return source.index(item)
            else:
                return -1
        return [index(item) for item in target]

    def _report(self, inputs):
        comm = inputs.get('communicator')
        if comm:
            # Divide work between MPI nodes.
            size = comm.Get_size()
            rank = comm.Get_rank()
            if rank == 0:
                each = math.ceil(len(inputs['source']) / size)
                source = [inputs['source'][i*each:(i+1)*each] for i in range(size)]
                source.reverse()
            else:
                source = None
            source = comm.scatter(source, root=0)
        else:
            rank = 0
            source = inputs['source']
        isFlat = 'flat' in inputs['options']
        # Check options.
        if (not isFlat) != ('structured' in inputs['options']):
            raise ValueException("Options 'flat' and 'structured' are mutually exclusive")
        # Check the that filenames in source are unique.
        basenames = [os.path.basename(path) for path in source]
        if len(basenames) != len(set(basenames)):
            raise ValueException("The basenames in source are not unique")
        # Check keys to unravel.
        for key in inputs['unravel_keys']:
            if len(inputs['parameters'][key]) != len(basenames):
                raise ValueException("Lengths of source and %s do not match" % key)
        # Run the analysis.
        chTables = salto.channelTables
        pm = salto.pluginManager
        tableName = salto.makeUniqueKey(salto.channelTables, 'temp')
        reports = {}
        for i, (path, basename) in enumerate(zip(source, basenames)):
            print(basename)
            chTables[tableName] = salto.ChannelTable()
            pm.read(path, inputs['source_format'], tableName)
            parameters = inputs['parameters'].copy()
            parameters['channelTable'] = tableName
            for key in inputs['unravel_keys']:
                parameters[key] = inputs['parameters'][key][i]
            result = pm.compute(inputs['analysis'], parameters)
            tmpTable = result.get('channelTable')
            if tmpTable:
                chTables.pop(tmpTable, None)
            if inputs['result_format'].lower() == 'html':
                reports[basename] = result['html']
            else:
                reports[basename] = result['table']
        chTables.pop(tableName, None)
        if comm:
            reports = comm.gather(reports, root=0)
            if rank == 0:
                reports = {k: v for d in reports for k, v in d.items()}
        # Create the report.
        if not comm or rank == 0:
            datestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            params = pm.computations[inputs['analysis']].computations[inputs['analysis']][1]
            preamble = [["analysis date", datestr],
                        ["analysis plugin", inputs['analysis']]]
            preamble += [[p[0], inputs['parameters'].get(p[0], p[3]), p[2]]
                         for p in params if p[0] != 'channelTable' and
                                            p[0] not in inputs['unravel_keys']]
            if inputs['result_format'].lower() == 'xlsx':
                wb = xlsxwriter.Workbook(inputs['result_file_path'],
                                         {'constant_memory': True,
                                          'nan_inf_to_errors': True,
                                          'default_date_format': 'yyyy-dd-mm hh:mm:ss'})
                ws = wb.add_worksheet('settings')
                bold = wb.add_format({'bold': 1})
                width = 8
                row = 0
                for row_data in preamble:
                    ws.write_row(row, 0, row_data)
                    row += 1
                    if len(row_data[0]) > width:
                        width = len(row_data[0])
                ws.set_column('A:A', width)
                if isFlat:
                    ws = wb.add_worksheet('results')
                    row = 0
                    headers = [reports[k]['header'] for k in reports
                               if 'header' in reports[k]]
                    if headers and headers.count(headers[0]) == len(headers):
                        # All headers are equal.
                        header = ['Filename'] + headers[0]
                        ws.write_row(row, 0, header, bold)
                        row += 1
                        rearrangeColumns = False
                    elif len(headers) == len(reports):
                        # Headers are not equal.
                        header = ['Filename'] + self._mergeHeaders(headers)
                        ws.write_row(row, 0, header, bold)
                        row += 1
                        rearrangeColumns = True
                    else:
                        raise ValueError("Header is required if all headers are not equal")
                    widths = ColumnWidths(header)
                    for basename, report in reports.items():
                        if rearrangeColumns:
                            order = self._columnOrder(report['header'],
                                                      header[1:])
                        for row_raw in report['data']:
                            if rearrangeColumns:
                                row_data = [(row_raw + [None])[i] for i in order]
                            else:
                                row_data = row_raw
                            ws.write_row(row, 0, [basename] + row_data)
                            row += 1
                            widths.update([basename] + row_data)
                    # Set column widths.
                    for i, w in enumerate(widths.widths):
                        ws.set_column(i, i, w)
                else:
                    for basename, report in reports.items():
                        ws = wb.add_worksheet(basename)
                        row = 0
                        if 'header' in report:
                            ws.write_row(row, 0, report['header'], bold)
                            widths = ColumnWidths(report['header'])
                            row += 1
                        else:
                            widths = ColumnWidths(report['data'][0])
                        for row_data in report['data']:
                            ws.write_row(row, 0, row_data)
                            widths.update(row_data)
                            row += 1
                        for row_data in report.get('summary', []):
                            ws.write_row(row, 0, row_data, bold)
                            widths.update(row_data)
                            row += 1
                        # Set column widths.
                        for i, w in enumerate(widths.widths):
                            ws.set_column(i, i, w)
                wb.close()
            elif inputs['result_format'].lower() == 'csv':
                with open(inputs['result_file_path'], 'w') as f:
                    writer = csv.writer(f)
                    # TODO: Add preamble (separate file?)
                    headers = [reports[k]['header'] for k in reports
                               if 'header' in reports[k]]
                    if headers and headers.count(headers[0]) == len(headers):
                        # All headers are equal.
                        header = ['Filename'] + headers[0]
                        rearrangeColumns = False
                    elif len(headers) == len(reports):
                        # Headers are not equal.
                        header = ['Filename'] + self._mergeHeaders(headers)
                        rearrangeColumns = True
                    else:
                        raise ValueError("Header is required if all headers "
                                         "are not equal.")
                    writer.writerow(header)
                    for basename, report in reports.items():
                        if rearrangeColumns:
                            order = self._columnOrder(report['header'],
                                                      header[1:])
                        for row_raw in report['data']:
                            if rearrangeColumns:
                                row_data = [(row_raw + [None])[i] for i in order]
                            else:
                                row_data = row_raw
                            writer.writerow([basename] + row_data)
            elif inputs['result_format'].lower() == 'html':
                raise NotImplementedError("HTML reporting is not implemented yet")
            else:
                raise ValueError("Unknown result format %s" % inputs['result_format'])
        return {}
