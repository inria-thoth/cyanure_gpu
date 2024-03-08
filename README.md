# Cyanure Pytorch

This project is a partial reimplementation of [Cyanure](https://github.com/inria-thoth/cyanure)


This is version is an implementation using python as the programming language as pytorch for the linear algebra part.

To be able to fit as much data as possible on a GPU which as much less memory than the RAM a server can have, the computation use floating point 32 numbers compared to the Cyanure CPU implementation which is using 64 bits floating point numbers for numerical stability.

The available solvers are ista and fista which can be used with catalyst or qning acceleration. 

The regularizations available are l1 and l2. The losses are square and logistic for binary and multiclass problem.


Installation from source
========================

Compared to the CPU implementation, to install it from source you just need to do 

> pip install .

from inside the project folder.

Create a new release
====================

When you wish to create a new version of the library you should open a merge 
request to merge on the master branch.

You should update the version of the library by incrementing the number version
in the __VERSION__ file.
Major version is dedicated to breaking changes.
Minor version to new features.
Fixes version to bug fixes release.

You should also update the __CHANGELOG__ file to pinpoint the modifications impacting the users.
