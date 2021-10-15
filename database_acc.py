# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:03:50 2021

@author: Elijah_Nkuah
"""
# DB
import sqlite3
conn = sqlite3.connect('bankaccount.db', check_same_thread=False)
c = conn.cursor()

# Functions

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data



def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data