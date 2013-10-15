package org.lemming.utils;

public class LemMING {
	public static void error(String errMessage) {
		System.err.println(errMessage);
		
		throw new RuntimeException(errMessage);
	}
}
