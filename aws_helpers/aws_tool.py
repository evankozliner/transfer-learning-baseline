#!/usr/bin/python
"""
    A tool for using EC2 a little more efficiently. 
"""

from boto import ec2
import os
import sys
import json
import time
import subprocess

def main():
    conn = get_ec2_connection()
    reservations = conn.get_all_instances()
    instances = [i for r in reservations for i in r.instances]
    instance = find_instance_by_nametag(instances, config("instance-name")) 
    handle_command(instance)

def spinlock_until(instance, state):
    """ Spinlocks the process until the EC2 instance is at the desired state """
    current_state = instance.state
    while current_state != state:
        print "Instance is at state: " + current_state + "..."
        time.sleep(5)
        current_state = instance.update()

def handle_command(instance):
    """ Takes action based on input string:
        start: starts the AWS GPU instance, polls until up
        stop: stops the AWS GPU instance, polls until done
        ip: returns the IP of the AWS instance
        """

    if len(sys.argv) < 2: sys.exit("Needs Argument Error: Usage 'python aws_tool.py *command*'") 
    if "start" in sys.argv[1]:
        if instance.state == "running":
            print "Instance is already running"
        else:
            print "Starting instance " + config("instance-name") + "..."
            instance.start()
            spinlock_until(instance, "running")
            print "Instance " + config("instance-name") + " is up on IP " + instance.ip_address
    elif "stop" in sys.argv[1]:
        if instance.state == "stopped":
            print "Instance is already stopped"
        else:
            print "Stopping instance " + config("instance-name") + "..."
            instance.stop()
            spinlock_until(instance, "stopped")
            print "Instance is stopped!"
    elif "ip" == sys.argv[1]:
        if instance.ip_address is None:
            print "Instance is " + instance.state + " so there is no associated IP."
        else:
            print instance.ip_address
    elif "ssh" == sys.argv[1]:
        err_msg = "Instance is not currently running. Run 'python aws_tool.py start' to run the instance"
        if instance.state != "running": sys.exit(err_msg)
        print "Attempting to log into host for user: " + config('ec2-user')
        cmd = "ssh -v " + config('ec2-user') + "@" + instance.ip_address
        retcode = subprocess.call(cmd, shell=True)
    else:
        print "Usage 'python aws_tool.py *command*'"
        print "*command* is one of these: 'start', 'stop', 'ip', 'ssh'"

def find_instance_by_nametag(instances, name):
    """ Returns an instance based on its name. """
    for i in instances:
        if "Name" in i.tags and name in i.tags['Name']:
            return i
    sys.exit("Sorry, I couldn't find an instance with that name!")

def get_ec2_connection():
    """ Returns and EC2 connection based on the environment variables configured in the aws_config file. """
    access = os.environ[config("access-environment-var")]
    secret= os.environ[config("secret-environment-var")]
    return ec2.connect_to_region(config("region"), 
            aws_access_key_id=access, aws_secret_access_key=secret)

# TODO get related path based on the python file
def config(key):
    """ Returns a string key from the aws_config json file"""
    with open("aws_config.json") as conf:
        return json.load(conf)[key]

if __name__ == "__main__":
    main()
