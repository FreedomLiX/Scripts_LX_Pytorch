import xml.dom
import xml.dom.minidom
import os
import cv2

_ROOT_NODE = 'annotation'
_DATABASE_NODE = 'Unknown'
_SEGMENTED_NODE = '0'
_POSE_NODE = 'Unspecified'
_TRUNCATED_NODE = '0'
_DIFFICULT_NODE = '0'
_FILE_NODE = 'None'


# 封装添加一个子节点
def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)
    parent_node.appendChild(child_node)


# 创建一个元素节点
def createElementNode(doc, tag, attr):
    element_node = doc.createElement(tag)
    text_node = doc.createTextNode(attr)  # 创建一个文本节点
    element_node.appendChild(text_node)   # 将文本节点作为元素节点的子节点
    return element_node


# 写入文件
def write_xml(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent='\t', newl='\n', encoding='utf-8')
    tmpfile.close()

    # 删除自动生成的首行文字
    fin = open('tmp.xml')
    fout = open(filename, 'w')
    lines = fin.readlines()
    for line in lines[1:]:
        if line.split():
            fout.writelines(line)
    fin.close()
    fout.close()
    os.remove('tmp.xml')


# 创建xml文件
def create_xml(idx, image, budboxes_labels, out_path, image_file):
    dom = xml.dom.getDOMImplementation()
    doc = dom.createDocument(None, _ROOT_NODE, None)

    root_node = doc.documentElement  # 获取根节点annotation
    createChildNode(doc, 'folder', _FILE_NODE, root_node)          # 子节点folder
    createChildNode(doc, 'filename', _FILE_NODE, root_node)        # 子节点filename
    createChildNode(doc, 'path', _FILE_NODE, root_node)            # 子节点path
    source_node = doc.createElement('source')                      # 子节点source
    createChildNode(doc, 'database', _DATABASE_NODE, source_node)  # 二级子节点database
    root_node.appendChild(source_node)

    size_node = doc.createElement('size')                            # 子节点size
    createChildNode(doc, 'width', str(image.shape[1]), size_node)    # 二级子节点width
    createChildNode(doc, 'height', str(image.shape[0]), size_node)   # 二级子节点height
    createChildNode(doc, 'depth', str(image.shape[2]), size_node)    # 二级子节点depth
    root_node.appendChild(size_node)
    createChildNode(doc, 'segmented', _SEGMENTED_NODE, root_node)     # 子节点segmented
    for budbox_label in budboxes_labels:                              # 每一个box都创建一个object节点
        label = budbox_label[-1]
        object_node = doc.createElement('object')                        # 子节点object
        createChildNode(doc, 'name', str(label), object_node)            # 二级子节点name
        createChildNode(doc, 'pose', _POSE_NODE, object_node)            # 二级子节点pose
        createChildNode(doc, 'truncated', _TRUNCATED_NODE, object_node)  # 二级子节点truncated
        createChildNode(doc, 'difficult', _DIFFICULT_NODE, object_node)  # 二级子节点difficult
        bndbox_node = doc.createElement('bndbox')                        # 二级子节点bndbox
        createChildNode(doc, 'xmin', str(int(budbox_label[0])), bndbox_node)
        createChildNode(doc, 'ymin', str(int(budbox_label[1])), bndbox_node)
        createChildNode(doc, 'xmax', str(int(budbox_label[0] + budbox_label[2])), bndbox_node)
        createChildNode(doc, 'ymax', str(int(budbox_label[1] + budbox_label[3])), bndbox_node)
        object_node.appendChild(bndbox_node)
        root_node.appendChild(object_node)
    # 写xml文件
    xml_path = os.path.join(out_path, 'Annotations')
    jpg_path = os.path.join(out_path, 'JPEGImages')
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    if not os.path.exists(jpg_path):
        os.makedirs(jpg_path)
    write_xml(doc, filename=xml_path + '/' + image_file[:-4] + '_'+'aug'+'_'+str(idx) + '.xml')
    # 写jpg文件
    cv2.imwrite(filename=jpg_path + '/' + image_file[:-4] + '_'+'aug'+'_'+str(idx) + '.jpg',
                img=image)
