#!/bin/bash

source `dirname "$0"`/vars.sh

helsinki_ini


#for BOOK in Hemingway; do
for BOOK in Auster Baldacci Barnes Boyne Carre Franzen French Golding Grisham Hemingway Highsmith Hosseini Irving James Joyce Kerouac King Kinsella Mitchell Orwell Patterson Pynchon Roth Rowling Salinger Slaughter Steinbeck Tolkien Twain Wilde Yalom; do
	python helsinki_ennl.py $BOOK
done
