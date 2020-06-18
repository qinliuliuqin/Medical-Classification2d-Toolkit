import copy
import os
import pandas as pd


def add_document_text(original_text, new_text_to_add):
    """
    Add document text file
    """
    return original_text + r'+"{0}"'.format(new_text_to_add)


def write_summary_csv_report_for_all_landmarks(error_summary, csv_file_path):
    """
    Write a html report for all landmarks to summary the detection results
    """
    summary = []
    for landmark_idx, landmark_name in enumerate(error_summary.all_cases.keys()):
        num_pos_cases = len(error_summary.tp_cases[landmark_name]) + \
                        len(error_summary.fn_cases[landmark_name])
        num_neg_cases = len(error_summary.tn_cases[landmark_name]) + \
                        len(error_summary.fp_cases[landmark_name])
        if len(error_summary.tp_cases[landmark_name]) == 0 and num_pos_cases == 0:
            tpr = 100
        else:
            tpr = len(error_summary.tp_cases[landmark_name]) / max(1, num_pos_cases) * 100
        if len(error_summary.tn_cases[landmark_name]) == 0 and num_neg_cases == 0:
            tnr = 100
        else:
            tnr = len(error_summary.tn_cases[landmark_name]) / max(1, num_neg_cases) * 100
        mean_error = error_summary.mean_error_tp[landmark_name]
        std_error = error_summary.std_error_tp[landmark_name]
        median_error = error_summary.median_error_tp[landmark_name]
        max_error = error_summary.max_error_tp[landmark_name]
        summary.append([landmark_idx, landmark_name, num_pos_cases, num_neg_cases,
                        tpr, tnr, mean_error, std_error, median_error, max_error])

    columns = ['landmark_idx', 'landmark_name', 'pos_cases', 'neg_cases', 'TPR (%)', 'TNR (%)',
               'mean error (mm)', 'stddev', 'median error (mm)', 'max error (mm)']
    df = pd.DataFrame(data=summary, columns=columns)
    df.to_csv(csv_file_path, index=False, float_format='%.2f')


def write_html_report(document_text, analysis_text, html_report_path, width):
    """
    Write the html report for a landmark.
    """
    f = open(html_report_path, 'w')
    message = """
    <html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
      <title>result analysis</title>
      <style type="text/css">
          *{
              padding:0;
              margin:0;
          }
          .content {
              width: %spx;
              z-index:2;
          }
          .content img {
              width: %spx;
              transition-duration:0.2s;
              z-index:1;
          }
          .content img:active {
              transform: scale(2);
              -webkit-transform: scale(2); /*Safari 和 Chrome*/
              -moz-transform: scale(2); /*Firefox*/
              -ms-transform: scale(2); /*IE9*/
              -o-transform: scale(2); /*Opera*/
          }
      </style>
    </head>
    <body>
      <h1> Summary:</h1>
      %s
      <script type="text/javascript">
        document.write(%s)               
      </script>
    </body>
    </html>""" % (width, width, analysis_text, document_text)

    f.write(message)
    f.close()


# def add_three_images(document_text, image_link_template, image_folder, images, width):
#     """
#     Add three plane images to the document text file
#     """
#     for idx in range(3):
#         document_text += "\n"
#         image_info = r'<td>{0}</td>'.format(image_link_template.format(
#             os.path.join(image_folder, images[idx]), width))
#         document_text = add_document_text(document_text, image_info)
#     return document_text


def add_image(document_text, image_link_template, image_folder, image_name, width):
    """
    Add a image to the document text file
    :param document_text:
    :param image_link_template:
    :param image_folder:
    :param image_name:
    :param width:
    :return:
    """
    document_text += "\n"
    image_info = r'<td>{0}</td>'.format(image_link_template.format(os.path.join(image_folder, image_name), width))
    document_text = add_document_text(document_text, image_info)
    return document_text


def gen_row_for_html(usage_flag, image_link_template, error_info_template, document_text, image_name,
                     image_idx, error_summary, picture_folder='./pictures', width=200):
    """
    Generate a line of html text contents for labelled cases, in the usage of label checking.
    """
    case_info = r'<b>Case nunmber</b>:{0}: {1}'.format(image_idx, image_name)

    if usage_flag == 1:
        pass
    elif usage_flag == 2:
        pass
    else:
        raise ValueError('Unsupported flag type!')

    document_text = add_document_text(document_text, case_info)

    document_text += "\n"
    document_text = add_document_text(document_text, "<table border=1><tr>")
    document_text = add_image(document_text, image_link_template, picture_folder, image_name, width)
    if usage_flag == 2:
        pass
    document_text += "\n"
    document_text = add_document_text(document_text, r'</tr></table>')

    return document_text


def gen_html_report(objects_dict, usage_flag, output_folder):
    """
    Generate HTML report for object detection.
    :param objects_dict:
    :param usage_flag: 1 for label checking, 2 for error evaluation
    :param output_folder:
    :return:
    """

    labelled_objects_dict = objects_dict[0]

    if usage_flag == 2:
        detected_objects_dict = objects_dict[1]
        assert len(labelled_objects_dict.keys()) == len(detected_objects_dict.keys())

    document_text = r'"<h1>check annotations:</h1>"'
    for image_idx, image_name in enumerate(labelled_objects_dict.keys()):
        print("Processing image {}.".format(image_name))
        image_link_template = r"<div class='content'><img border=0  src= '{0}'  hspace=1  width={1} class='pic'></div>"
        error_info_template = r'<b>Labelled</b>: {};'
        document_text += "\n"

        if usage_flag == 1:
            document_text = gen_row_for_html(usage_flag,
                                             image_link_template,
                                             error_info_template,
                                             document_text,
                                             image_name,
                                             image_idx,
                                             None)

        elif usage_flag == 2:
            pass
        else:
            raise ValueError('Undefined usage flag!')

        if usage_flag == 1:
            pass
        elif usage_flag == 2:
            pass
        else:
            raise ValueError('Undefined usage float!')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    html_report_name = 'result_analysis.html'
    html_report_path = os.path.join(output_folder, html_report_name)
    write_html_report(document_text, '', html_report_path, width=200)