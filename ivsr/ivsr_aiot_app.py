#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from optparse import OptionParser
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path

perf_data = None
frame_count = {}
saved_count = {}
global PGIE_CLASS_ID_VEHICLE
PGIE_CLASS_ID_VEHICLE = 0
global PGIE_CLASS_ID_PERSON
PGIE_CLASS_ID_PERSON = 2

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]
schema_type = 0
#msgbroker property
proto_lib = "/opt/nvidia/deepstream/deepstream-6.3/lib/libnvds_amqp_proto.so"
conn_str = "192.168.0.201;5672;admin;admin"
cfg_file = "cfg_amqp.txt"
topic = None
no_display = False

PGIE_CONFIG_FILE = "dstest_imagedata_config.txt"
MSCONV_CONFIG_FILE = "dstest4_msgconv_config.txt"
MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4

def meta_copy_func(data, user_data):
    # Cast data to pyds.NvDsUserMeta
    user_meta = pyds.NvDsUserMeta.cast(data)
    src_meta_data = user_meta.user_meta_data
    # Cast src_meta_data to pyds.NvDsEventMsgMeta
    srcmeta = pyds.NvDsEventMsgMeta.cast(src_meta_data)
    # Duplicate the memory contents of srcmeta to dstmeta
    # First use pyds.get_ptr() to get the C address of srcmeta, then
    # use pyds.memdup() to allocate dstmeta and copy srcmeta into it.
    # pyds.memdup returns C address of the allocated duplicate.
    dstmeta_ptr = pyds.memdup(pyds.get_ptr(srcmeta),
                              sys.getsizeof(pyds.NvDsEventMsgMeta))
    # Cast the duplicated memory to pyds.NvDsEventMsgMeta
    dstmeta = pyds.NvDsEventMsgMeta.cast(dstmeta_ptr)

    # Duplicate contents of ts field. Note that reading srcmeat.ts
    # returns its C address. This allows to memory operations to be
    # performed on it.
    dstmeta.ts = pyds.memdup(srcmeta.ts, MAX_TIME_STAMP_LEN + 1)

    # Copy the sensorStr. This field is a string property. The getter (read)
    # returns its C address. The setter (write) takes string as input,
    # allocates a string buffer and copies the input string into it.
    # pyds.get_string() takes C address of a string and returns the reference
    # to a string object and the assignment inside the binder copies content.
    dstmeta.sensorStr = pyds.get_string(srcmeta.sensorStr)

    if srcmeta.objSignature.size > 0:
        dstmeta.objSignature.signature = pyds.memdup(
            srcmeta.objSignature.signature, srcmeta.objSignature.size)
        dstmeta.objSignature.size = srcmeta.objSignature.size

    if srcmeta.extMsgSize > 0:
        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_VEHICLE:
            srcobj = pyds.NvDsVehicleObject.cast(srcmeta.extMsg)
            obj = pyds.alloc_nvds_vehicle_object()
            obj.type = pyds.get_string(srcobj.type)
            obj.make = pyds.get_string(srcobj.make)
            obj.model = pyds.get_string(srcobj.model)
            obj.color = pyds.get_string(srcobj.color)
            obj.license = pyds.get_string(srcobj.license)
            obj.region = pyds.get_string(srcobj.region)
            dstmeta.extMsg = obj
            dstmeta.extMsgSize = sys.getsizeof(pyds.NvDsVehicleObject)
        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON:
            srcobj = pyds.NvDsPersonObject.cast(srcmeta.extMsg)
            obj = pyds.alloc_nvds_person_object()
            obj.age = srcobj.age
            obj.gender = pyds.get_string(srcobj.gender)
            obj.cap = pyds.get_string(srcobj.cap)
            obj.hair = pyds.get_string(srcobj.hair)
            obj.apparel = pyds.get_string(srcobj.apparel)
            dstmeta.extMsg = obj
            dstmeta.extMsgSize = sys.getsizeof(pyds.NvDsVehicleObject)

    return dstmeta
# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
# Callback function for freeing an NvDsEventMsgMeta instance
def meta_free_func(data, user_data):
    user_meta = pyds.NvDsUserMeta.cast(data)
    srcmeta = pyds.NvDsEventMsgMeta.cast(user_meta.user_meta_data)

    # pyds.free_buffer takes C address of a buffer and frees the memory
    # It's a NOP if the address is NULL
    pyds.free_buffer(srcmeta.ts)
    pyds.free_buffer(srcmeta.sensorStr)

    if srcmeta.objSignature.size > 0:
        pyds.free_buffer(srcmeta.objSignature.signature)
        srcmeta.objSignature.size = 0

    if srcmeta.extMsgSize > 0:
        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_VEHICLE:
            obj = pyds.NvDsVehicleObject.cast(srcmeta.extMsg)
            pyds.free_buffer(obj.type)
            pyds.free_buffer(obj.color)
            pyds.free_buffer(obj.make)
            pyds.free_buffer(obj.model)
            pyds.free_buffer(obj.license)
            pyds.free_buffer(obj.region)
        if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON:
            obj = pyds.NvDsPersonObject.cast(srcmeta.extMsg)
            pyds.free_buffer(obj.gender)
            pyds.free_buffer(obj.cap)
            pyds.free_buffer(obj.hair)
            pyds.free_buffer(obj.apparel)
        pyds.free_gbuffer(srcmeta.extMsg)
        srcmeta.extMsgSize = 0


def generate_vehicle_meta(data):
    obj = pyds.NvDsVehicleObject.cast(data)
    obj.type = "sedan"
    obj.color = "blue"
    obj.make = "Bugatti"
    obj.model = "M"
    obj.license = "XX1234"
    obj.region = "CA"
    return obj


def generate_person_meta(data):
    obj = pyds.NvDsPersonObject.cast(data)
    obj.age = 45
    obj.cap = "none"
    obj.hair = "black"
    obj.gender = "male"
    obj.apparel = "formal"
    return obj


def generate_event_msg_meta(data, class_id):
    meta = pyds.NvDsEventMsgMeta.cast(data)
    meta.sensorId = 0
    meta.placeId = 0
    meta.moduleId = 0
    meta.sensorStr = "sensor-0"
    meta.ts = pyds.alloc_buffer(MAX_TIME_STAMP_LEN + 1)
    pyds.generate_ts_rfc3339(meta.ts, MAX_TIME_STAMP_LEN)

    # This demonstrates how to attach custom objects.
    # Any custom object as per requirement can be generated and attached
    # like NvDsVehicleObject / NvDsPersonObject. Then that object should
    # be handled in payload generator library (nvmsgconv.cpp) accordingly.
    if class_id == PGIE_CLASS_ID_VEHICLE:
        meta.type = pyds.NvDsEventType.NVDS_EVENT_MOVING
        meta.objType = pyds.NvDsObjectType.NVDS_OBJECT_TYPE_VEHICLE
        meta.objClassId = PGIE_CLASS_ID_VEHICLE
        obj = pyds.alloc_nvds_vehicle_object()
        obj = generate_vehicle_meta(obj)
        meta.extMsg = obj
        meta.extMsgSize = sys.getsizeof(pyds.NvDsVehicleObject)
    if class_id == PGIE_CLASS_ID_PERSON:
        meta.type = pyds.NvDsEventType.NVDS_EVENT_ENTRY
        meta.objType = pyds.NvDsObjectType.NVDS_OBJECT_TYPE_PERSON
        meta.objClassId = PGIE_CLASS_ID_PERSON
        obj = pyds.alloc_nvds_person_object()
        obj = generate_person_meta(obj)
        meta.extMsg = obj
        meta.extMsgSize = sys.getsizeof(pyds.NvDsPersonObject)
    return meta

def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            # Periodically check for objects with borderline confidence value that may be false positive detections.
            # If such detections are found, annotate the frame with bboxes and confidence value.
            # Save the annotated frame to file.
            if saved_count["stream_{}".format(frame_meta.pad_index)] % 30 == 0 and (
                    MIN_CONFIDENCE < obj_meta.confidence < MAX_CONFIDENCE):
                if is_first_obj:
                    is_first_obj = False
                    # Getting Image data using nvbufsurface
                    # the input should be address of buffer and batch_id
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
                    # convert python array into numpy array format in the copy mode.
                    frame_copy = np.array(n_frame, copy=True, order='C')
                    # convert the array into cv2 default color format
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                    if is_aarch64(): # If Jetson, since the buffer is mapped to CPU for retrieval, it must also be unmapped 
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) # The unmap call should be made after operations with the original array are complete.
                                                                                            #  The original array cannot be accessed after this call.

                save_image = True

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print("Frame Number=", frame_number, "Number of Objects=", num_rects, "Vehicle_count=",
              obj_counter[PGIE_CLASS_ID_VEHICLE], "Person_count=", obj_counter[PGIE_CLASS_ID_PERSON])
        # update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        if save_image:
            img_path = "{}/stream_{}/frame_{}.jpg".format(folder_name, frame_meta.pad_index, frame_number)
            cv2.imwrite(img_path, frame_copy)
        saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    # image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    color = (0, 0, 255, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 6)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name + ',C=' + str(confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255, 0), 2)
    return image


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if not is_aarch64() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        Object.set_property("cudadec-memtype", 2)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)

    global perf_data
    perf_data = PERF_DATA(len(args) - 2)
    number_sources = len(args) - 2

    global folder_name
    folder_name = args[-1]
    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)

    os.mkdir(folder_name)
    print("Frames will be saved in ", folder_name)
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        os.mkdir(folder_name + "/stream_" + str(i))
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    msgconv = Gst.ElementFactory.make("nvmsgconv", "nvmsg-converter")
    if not msgconv:
        sys.stderr.write(" Unable to create msgconv \n")

    msgbroker = Gst.ElementFactory.make("nvmsgbroker", "nvmsg-broker")
    if not msgbroker:
        sys.stderr.write(" Unable to create msgbroker \n")
    tee = Gst.ElementFactory.make("tee", "nvsink-tee")
    if not tee:
        sys.stderr.write(" Unable to create tee \n")

    queue1 = Gst.ElementFactory.make("queue", "nvtee-que1")
    if not queue1:
        sys.stderr.write(" Unable to create queue1 \n")

    queue2 = Gst.ElementFactory.make("queue", "nvtee-que2")
    if not queue2:
        sys.stderr.write(" Unable to create queue2 \n")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if no_display:
        print("Creating FakeSink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if not sink:
            sys.stderr.write(" Unable to create fakesink \n")
    else:
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")
    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "dstest_imagedata_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
              number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    msgconv.set_property('config', MSCONV_CONFIG_FILE)
    msgconv.set_property('payload-type', schema_type)
    msgbroker.set_property('proto-lib', proto_lib)
    msgbroker.set_property('conn-str', conn_str)
    if cfg_file is not None:
        msgbroker.set_property('config', cfg_file)
    if topic is not None:
        msgbroker.set_property('topic', topic)
    msgbroker.set_property('sync', False)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(msgconv)
    pipeline.add(msgbroker)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)
    queue1.link(msgconv)
    msgconv.link(msgbroker)
    queue2.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if i != 0:
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
