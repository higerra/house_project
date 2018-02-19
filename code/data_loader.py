import argparse
import os
import shutil
import urllib.request, urllib.error
import xml.etree.ElementTree as ET


def process_folder(base_url, folder_url, xmlns, local_base, reload=False):
    list_url = base_url + '?list-type=2&delimiter=/&prefix=' + folder_url
    with urllib.request.urlopen(list_url) as list_response:
        # Make sure corresponding local folder exists
        local_folder = local_base + '/' + folder_url
        if not os.path.exists(local_folder):
            os.makedirs(local_folder, exist_ok=True)
        list_xml = ET.fromstring(list_response.read())
        # First save files from this folder, if any
        for file_item in list_xml.findall('{%s}Contents' % xmlns):
            for path in file_item.findall('{%s}Key' % xmlns):
                print('Downloading ' + path.text)
                remote_path = base_url + '/' + path.text
                local_path = local_base + '/' + path.text
                if os.path.exists(local_path) and not reload:
                    print('File exist, skip')
                    continue
                with urllib.request.urlopen(remote_path) as path_response, open(local_path, 'wb') as f:
                    shutil.copyfileobj(path_response, f)

        for folder_item in list_xml.findall('{%s}CommonPrefixes' % xmlns):
            for path in folder_item.findall('{%s}Prefix' % xmlns):
                process_folder(base_url, path.text, xmlns, local_base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xmlns', type=str, default='http://s3.amazonaws.com/doc/2006-03-01/')
    parser.add_argument('--base_url', type=str, default='https://s3-us-west-2.amazonaws.com/renoworks-all-projects')
    parser.add_argument('--source_dir', type=str, default='projects/Cuttlefish/uploaded/homeplay/projects/')
    parser.add_argument('--local_base', type=str, default='../renoworks')
    parser.add_argument('--reload', action='store_true')

    args = parser.parse_args()

    process_folder(args.base_url, args.source_dir, args.xmlns, args.local_base, args.reload)
