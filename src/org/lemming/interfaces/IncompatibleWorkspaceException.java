package org.lemming.interfaces;

public class IncompatibleWorkspaceException extends Exception {
	public IncompatibleWorkspaceException(String missingMember) {
		super("Member "+missingMember+" is missing from the workspace");
	}
}
