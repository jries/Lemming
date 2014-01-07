package org.lemming.interfaces;

import org.lemming.data.Workspace;

/**
 * The WorkspacePlugin interface provides a plugin interface for modules to be used in a pipeline using a Workspace as input and output. 
 * It also contains the constants needed to specify the required fields within the standard fields of the workspace (X,Y,Z,frame,channel and ID).
 * 
 * @author Thomas Pengo
 *
 */

public interface WorkspacePlugin {
	public static final int NEEDS_X = 1;
	public static final int NEEDS_Y = 2;
	public static final int NEEDS_Z = 4;
	public static final int NEEDS_FRAME = 8;
	public static final int NEEDS_CHAN = 16;
	public static final int NEEDS_ID = 32;
	
	public static final String[] NO_REQUIREMENTS = new String[] {};

	/**
	 * Checks requirements and if the Workspace is compatible, then sets the input accordingly.
	 * 
	 * @param in the input Workspace
	 * @throws IncompatibleWorkspaceException if not compatible
	 */
	public void setInput(Workspace in) throws IncompatibleWorkspaceException;
	
	/**
	 * Sets output to workspace out and creates appropriate members if necessary
	 * 
	 * @param out
	 */
	public void setOutput(Workspace out);

	/**
	 * Returns an array of Strings 
	 * @return
	 */
	public abstract String[] getRequiredMembers();

	public abstract void setRequiredMembers(int flags, String[] requiredMembers);

	public abstract void setRequiredMembers(int flags);	
}

