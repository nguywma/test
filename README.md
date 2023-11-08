# AIoT-ds
Dependencies
-------------
* glib 2.0
----------
 sudo apt-get install libglib2.0 libglib2.0-dev
 
Install rabbitmq-c library
--------------------------
 sudo apt-get install librabbitmq-dev
 
 sudo cp /usr/lib/x86_64-linux-gnu/librabbitmq.so /opt/nvidia/deepstream/deepstream-6.3/lib #add libraly to deepstream 
 
 sudo cp /usr/lib/x86_64-linux-gnu/librabbitmq.so.4 /opt/nvidia/deepstream/deepstream-6.3/lib 
 
 sudo cp /usr/lib/x86_64-linux-gnu/librabbitmq.so.4.4.0 /opt/nvidia/deepstream/deepstream-6.3/lib #add libraly to deepstream 
 
 sudo cp /usr/lib/x86_64-linux-gnu/librabbitmq.a /opt/nvidia/deepstream/deepstream-6.3/lib #add libraly to deepstream 
 
If you plan to have AMQP broker installed on your local machine
---------------------------------------------------------------
#Install rabbitmq on your ubuntu system: https://www.rabbitmq.com/install-debian.html

 sudo apt-get install rabbitmq-server

#Ensure rabbitmq service has started by running (should be the case):

 sudo service rabbitmq-server status

#Otherwise

 sudo service rabbitmq-server start
 
Create exchange , queue & and bind queue to exchange:
-----------------------------------------------------

# Rabbitmq management:

It comes with a command line tool which you can use to create/configure all of your queues/exchanges/etc
https://www.rabbitmq.com/management.html

# Install rabbitmq management plugin:

sudo rabbitmq-plugins enable rabbitmq_management

# Use the default exchange amq.topic

OR create an exchange as below, the same name as the one you specify within the cfg_amqp.txt
#sudo rabbitmqadmin -u guest -p guest -V / declare exchange name=myexchange type=topic

# Create a Queue

sudo rabbitmqadmin -u guest -p guest -V / declare queue name=myqueue durable=false auto_delete=true

# BIND QUEUE TO EXCHANGE WITH ROUTHING_KEY SPECIFICATION (MANDANTORY) 
rabbitmqadmin -u guest -p guest -V / declare binding source=amq.topic destination=myqueue routing_key=topicname

Setup the message_converter_broker: 
-----------------------------------

#msg_broker_proto_lib: /opt/nvidia/deepstream/deepstream/lib/libnvds_amqp_proto.so

#msg_broker_conn_str: <host;port;username;password>

#msg_conv_config: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.txt

#topic: <topic_name>

#msg_broker_config: nvgraph/cfg_amqp.txt

sending message from localhost to cloud
----------------------------------------
$ cd cloud-based/python
$ python3 event_transfer.py 

