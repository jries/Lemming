package org.lemming.utils;

import java.lang.reflect.Field;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.lemming.interfaces.Parameter;

public class Parameters {

	/**
	 * Returns the parameter field of o.
	 * 
	 * @param o
	 * @param field
	 * @return
	 * @throws NoSuchFieldException
	 * @throws SecurityException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	public static Object getParameter(Object o, String field) throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		Field f = o.getClass().getField(field);
		if (f != null && f.isAnnotationPresent(Parameter.class)) {
			return f.get(o);
		} else
			throw new RuntimeException("Object "+o+" has no such parameter "+field);
	}

	public static Map<String,Object> getParameterMap(final Object o) {
		return new Map<String,Object>() {
			/**
			 * Does not do anything.
			 */
			@Override
			public void clear() {
				// Does not do anything
			}
			
			/**
			 * Checks if that parameter exists.
			 */
			@Override
			public boolean containsKey(Object key) {
				try {
					Field f = o.getClass().getField(key.toString());
					
					return f!=null && f.isAnnotationPresent(Parameter.class);
				} catch(Exception e) {
					e.printStackTrace();
					return false;
				}
			}
			
			/**
			 * Checks if among the parameters, any has that value.
			 * 
			 */
			@Override
			public boolean containsValue(Object value) {
				try {
					for (Field f : o.getClass().getFields()) {
						if (value.equals(f.get(o)))
							return true;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				return false;
			}
			
			/**
			 * NOT IMPLEMENTED.
			 */
			@Override
			public Set<java.util.Map.Entry<String, Object>> entrySet() {
				// TODO not implemented
				return null;
			}
			
			
			/**
			 * Returns the specified parameter.
			 */
			@Override
			public Object get(Object key) {
				try {
					Field f = o.getClass().getField(key.toString());
					
					if ( f!=null && f.isAnnotationPresent(Parameter.class) )
						return f.get(o);
				} catch(Exception e) {
					e.printStackTrace();
					return null;
				}
				return null;
			}
			
			@Override
			public Set<String> keySet() {
				Set<String> set = new HashSet<String>();
				for (Field f : o.getClass().getFields()) {
					if (f.isAnnotationPresent(Parameter.class))
						set.add(f.getName());
				}
				return set;
			}
			
			@Override
			public Object put(String key, Object value) {
				try {
					Field f = o.getClass().getField(key.toString());
					
					if ( f!=null && f.isAnnotationPresent(Parameter.class) ) {
						f.set(o, value);
						return value;
					}
				} catch(Exception e) {
					e.printStackTrace();
					return null;
				}
				return null;
			}
			
			@Override
			public void putAll(Map<? extends String, ? extends Object> m) {
				for (Entry<? extends String,? extends Object> e : m.entrySet())
					put(e.getKey(),e.getValue());
			}
			
			/**
			 * Not implemented.
			 */
			@Override
			public Object remove(Object key) {
				return null;
			}
			
			/**
			 * Returns the number of parameters.
			 */
			@Override
			public int size() {
				return keySet().size();
			}
			
			/**
			 * Returns all values. Not implemented.
			 */
			@Override
			public Collection<Object> values() {
				// TODO not implemented
				return null;
			}
			
			/**
			 * Returns true if no parameter has been found.
			 */
			@Override
			public boolean isEmpty() {
				return keySet().isEmpty();
			}
		};
	}

	/**
	 * Sets the parameter field of o to value.
	 * 
	 * @param o
	 * @param field
	 * @param value
	 * @throws NoSuchFieldException
	 * @throws SecurityException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	public static void setParameter(Object o, String field, Object value) throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		Field f = o.getClass().getField(field);
		if (f != null && f.isAnnotationPresent(Parameter.class)) {
			f.set(o, value);
		} else
			throw new RuntimeException("Object "+o+" has no such parameter "+field);
	}

}
