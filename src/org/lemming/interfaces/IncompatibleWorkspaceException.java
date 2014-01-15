package org.lemming.interfaces;

/**
 * Launched from when trying to link a workspace plugin to an incompatible Workspaces.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class IncompatibleWorkspaceException extends Exception {
	public IncompatibleWorkspaceException(String missingMember) {
		super("Member "+missingMember+" is missing from the workspace");
	}
}
