package org.lemming.interfaces;

import java.util.List;

//import java.util.List;

/**
 * Represents a table of generic localizations. Each row is a localization and each column represents
 * a different parameter of that localization. It can be accessed with an iterator.
 * 
 * Each column is called 'member'.
 * 
 * There are six 'special' member. X,Y,Z,channel,frame and ID. The name of each of these member can 
 * be anything, and is remembered in the workspace, so that the getRow() will return a GenericLocalization
 * with the data collected from the appropriate member.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface Workspace extends Iterable<GenericLocalization> {

	/**
	 * @param member - the specified member
	 * @return Return the entire column as present in the workspace at the time the method is called.
	 */
	public List<Object> getMember(String member);
	
	/**
	 * @param member - the specified member
	 * @return Checks if the specified member is present in the workspace.
	 */
	public boolean hasMember(String member);
	
	/**
	 * Creates a new member in the workspace, with empty member in order to keep all member
	 * of the same length (table rectangular).
	 * 
	 * @param member - the specified member
	 */
	public void addNewMember(String member);
	
	/**
	 * @param row - row
	 * @return Return a particular row in the table.
	 */
	public GenericLocalization getRow(int row);
	
	/**
	 * Add a new row to the table. If the localization does not have a particular member, 
	 * a default value (null) will be inserted into the workspace. If the workspace does not have
	 * a particular member present in g, that member will be ignored.
	 * 
	 * @param g - GenericLocalization 
	 */
	public void addRow(GenericLocalization g);
	
	/**
	 * @param row - Removes the specified row.
	 */
	public void deleteRow(int row);
	
	/**
	 * Copy all rows from "from" to "to" into the current workspace.
	 * 
	 * @param g - Workspace
	 * @param from - start
	 * @param to - end
	 */
	public void addRows(Workspace g, int from, int to);
	
	/**
	 * Copy all rows from g to the current workspace.
	 * 
	 * @param g - workspace
	 */
	public void addAll(Workspace g);
	
	/**
	 * @param name - name to set
	 */
	public void setXname(String name);
	/**
	 * @param name - name to set
	 */
	public void setYname(String name);
	/**
	 * @param name - name to set
	 */
	public void setZname(String name);
	/**
	 * @param name - name to set
	 */
	public void setChannelName(String name);
	/**
	 * @param name - name to set
	 */
	public void setFrameName(String name);
	/**
	 * @param name - name to set
	 */
	public void setIDname(String name);
	
	/**
	 * @return name
	 */
	public String getXname();
	/**
	 * @return name
	 */
	public String getYname();
	/**
	 * @return name
	 */
	public String getZname();
	/**
	 * @return name
	 */
	public String getChannelName();
	/**
	 * @return name
	 */
	public String getFrameName();
	/**
	 * @return name
	 */
	public String getIDname();
	
	/**
	 * @return name
	 */
	public GenericLocalization newRow();

	/**
	 * @return Returns the number of rows in the table.
	 */
	int getNumberOfRows();

}
